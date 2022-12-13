library("stars")
tab = read.csv2("Tabela.csv", fileEncoding = "cp1250")

files = list.files("DIO.7211.342.2021_12.01.22_HSz", recursive = TRUE,
                   pattern = "OK_POWIERZCHNIE_PODSTAWOWE+\\.shp$",
                   full.names = TRUE)

getRegion = function(f) {
  x = unlist(strsplit(f, split = "/", fixed = TRUE))
  x = x[2]
  x = unlist(strsplit(x, "_", fixed = TRUE))
  x = x[-c(1:5)]
  x = x[1:(length(x) - 1)] # remove last segment
  x = paste(x, collapse = "_")
  return(x)
}

res = c(30, 30)
dir.create("reference")

for (f in files) {

  vec = read_sf(f)
  vec = st_transform(vec, crs = 2180)
  vec = merge(vec, tab[, c(1, 4)], by = "ID_FORM")
  name = getRegion(f)
  template = st_as_stars(st_bbox(vec), dx = res[1], dy = res[2], values = 0L)
  ras = st_rasterize(vec["CODE"], template, proxy = TRUE)
  filepath = paste0("reference/", name, ".tif")
  write_stars(ras, filepath, type = "Byte", NA_value = 0)

}


### create data frame from rasters ###
files = list.files("reference", pattern = ".tif", full.names = TRUE)
variables = list.files("variables", pattern = ".tif", full.names = TRUE)
var = read_stars(variables, proxy = TRUE)

class_vec = integer()
names = c("elevation", "slope", "stdev", "multitpi", "convexity",
           "entropy", "openness", "median500", "median1000")
var_df = data.frame()

for (f in files) {

  class = read_stars(f, proxy = FALSE)
  class[[1]] = as.integer(class[[1]])
  var_crop = st_crop(var, st_bbox(class))
  var_crop = st_as_stars(var_crop)
  var_crop = st_warp(var_crop, class, method = "near") # align grids
  class = as.vector(class[[1]])
  var_crop = data.frame(var_crop)[, -c(1, 2)]
  colnames(var_crop) = names

  class_vec = c(class_vec, class)
  var_df = rbind(var_df, var_crop)

}

rm(var_crop, class)
df = cbind(class = factor(class_vec), var_df) # ID class as factor
df = df[which(complete.cases(df)), ] # this is faster than `na.omit`
rm(var_df, class_vec)

## deal with class inbalance
## optionally, we can add weights to the classes
tab = sort(table(df$class))
tab

## reduce largest classes to 150k observations
size = 150000
cols_idx = names(which(tab > size))

for (i in cols_idx) {
  val = tab[[i]] - size
  idx = sample(which(df$class == i), val)
  df[idx, ] = NA
}
rm(idx)

## remove smallest classes: 50, 21
df[which(df$class == 50), ] = NA
df[which(df$class == 21), ] = NA

df = df[which(complete.cases(df)), ]
df$class = droplevels(df$class)

saveRDS(df, "dataset.rds")

### machine learning ###
library("corrr")
library("ranger")
library("yardstick")

set.seed(1)
n = round(0.7 * nrow(df))
trainIndex = sample(nrow(df), size = n)
train = df[trainIndex, ]
test = df[-trainIndex, ]
rm(df, trainIndex)

cor = correlate(train[, -1])
rplot(cor, print_cor = TRUE)

train = train[1:1200000, ] ### limit train data
test = test[1:500000, ] ### limit test data

rf_mdl = ranger(class ~ ., train, importance = "impurity", mtry = 3,
                seed = 1)
rf_mdl$predictions = NULL # remove predictions from model
rf_mdl$prediction.error #> 0.10996

pred = predict(rf_mdl, test[, -1])$predictions
accuracy_vec(test$class, pred) #> 0.893332
kap_vec(test$class, pred) #> 0.882949
mcc_vec(test$class, pred) #> 0.8833784

# importance plot
barplot(sort(importance(rf_mdl)))

saveRDS(rf_mdl, "rf_mdl.rds")

### predict on map ###
files = list.files("reference", pattern = ".tif", full.names = TRUE)
variables = list.files("variables", pattern = ".tif", full.names = TRUE)
var = read_stars(variables, proxy = TRUE)
rf_mdl = readRDS("rf_mdl.rds")
names = c("elevation", "slope", "stdev", "multitpi", "convexity",
          "entropy", "openness", "median500", "median1000")

n = 1 # number of sheet
r = read_stars(files[n])
var_crop = st_crop(var, st_bbox(r))
df = as.data.frame(var_crop)
crds = df[, 1:2]
df = df[, -c(1, 2)]
colnames(df) = names
idx = which(complete.cases(df))
pred = predict(rf_mdl, df[idx, ])$predictions
pred = as.integer(as.character(pred)) # make sure the class ID will be the same
crds = cbind(crds, class = NA)
crds[idx, 3] = pred

output = st_as_stars(crds)
st_crs(output) = 2180

write_stars(output, "predict.tif", type = "Byte", NA_value = 0)
