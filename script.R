# Install packages and dependencies if not already installed
packages <- c("caret", "dplyr", "mlbench", "reshape2")

for (pkg in packages) {
  if (!(pkg %in% rownames(installed.packages()))){
    install.packages(pkg, dependencies=TRUE)
    print(paste("Installing package ", pkg))
  }
  if (!(pkg %in% rownames(.packages()))) {
    print(paste("Loading package", pkg))
    library(pkg, character.only = TRUE)
  }
}

# Check NA values
check_na_inf_presence <- function(dataframe) {
  any_na_infinite = apply(dataframe, 2, function(x) any(is.na(x) | is.infinite(x)))
  # Will return TRUE if there is any na or inf value, FALSE otherwisse
  return(any(any_na_infinite))
}

###### DECLARE FUNCTIONS ######

plot_data_distribution <- function(data_df, y_col_name, main_title, x_label, y_label, filename=NULL){
  plot <- ggplot(data.frame(data_df[[y_col_name]]), aes(x=data_df[[y_col_name]])) +
    geom_bar() + 
    geom_text(stat='count', aes(label=..count..), vjust=-0.5)  +
    labs(x=x_label, y=y_label) + 
    ggtitle(main_title) + 
    theme(plot.title = element_text(hjust = 0.5))
  
  print(plot)
  
  if (!is.null(filename)) {
    dev.copy(png, filename, width=6, height=6, units="in",res=500)
    dev.off()
  }
}

# Plot confusion matrix function
plot_confusion_matrix <- function(confusion_matrix, filename=NULL) {
  confusion_matrix_df <- data.frame(confusion_matrix['table'])
  # Reverse Y-axis classes (Desc Top to Bottom)
  confusion_matrix_df <- confusion_matrix_df %>%
    mutate(x = factor(table.Reference),
           y = factor(table.Prediction, levels=rev(unique(table.Prediction)))) 
  
  plot <- ggplot(confusion_matrix_df, aes(x=x, y=y, fill=table.Freq)) +
    geom_tile(color="black") + theme_bw() + coord_equal() +
    scale_fill_distiller(palette="Greens", direction=1) +
    guides(fill=F) + # Remove legend
    labs(title="Confusion Matrix", x="Actual", y="Predicted") + 
    geom_text(aes(label=table.Freq), color="black")
  
  print(plot)
  
  if (!is.null(filename)) {
    dev.copy(png, filename, width=6, height=6, units="in",res=500)
    dev.off()
  }
}

# Plot function for feature correlation heatmap
plot_feature_correlation_heatmap <- function(data_df, filename=NULL) {
  corr_matrix <- round(cor(data_df[, -ncol(data_df)]),2)
  
  # Get only upper-triangular values
  corr_matrix[upper.tri(corr_matrix)] <- NA
  melted_corr_matrix <- melt(corr_matrix)
  
  plot <- ggplot(data = melted_corr_matrix, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1,1), space = "Lab", 
                         name="Pearson\nCorrelation") +
    theme_minimal()+ 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                     size = 12, hjust = 1))+
    coord_fixed() + 
    geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank())
  
  print(plot)
  
  if (!is.null(filename)) {
    dev.copy(png, filename, width=6, height=6, units="in",res=500)
    dev.off()
  }
}

plot_boxplot <- function(data_df, title, filename=NULL){
  boxplot(data_df, main=title)

  if (!is.null(filename)) {
    dev.copy(png, filename, width=6, height=6, units="in",res=500)
    dev.off()
  }
}

argmax <- function(vector){
  max <- NULL
  indices <- 1:length(vector)
  max_idx <- NULL
  for (i in indices) {
    val <- vector[i]
    if ((is.null(max)) || (val > max)) {
      max_idx <- i
      max <- val
    }
  }
  return(max_idx)
}

plot_tree_tuning <- function(results, num_trees, filename=NULL){
  tune_trees_df <- data.frame(Num_trees=num_trees, Accuracy=results)
  plot <- ggplot(tune_trees_df, aes(x=Num_trees, y=Accuracy)) + 
    geom_line() +
    labs(title="Number of Trees Hyperparameter Tuning", x="Number of Trees", y="Accuracy")
  
  print(plot)
  
  if (!is.null(filename)) {
    dev.copy(png, filename, width=6, height=6, units="in",res=500)
    dev.off()
  }
}

tune_num_trees <- function(min_trees, max_trees) {
  results <- c()
  models <- c()
  num_trees <- seq(min_trees, max_trees, by=100)
  
  for (i in num_trees) {
    model_rf <- train(Type~., data=train_df, method='rf', metric='Accuracy', ntree=i, trControl=trainCtrl)
    models <- append(models, model_rf)
    results <- append(results, max(model_rf$results$Accuracy))
  }
  
  best_idx <- argmax(results)
  best_num_trees <- num_trees[best_idx]
  best_model <- models[best_idx]
  output <- list("bestModel"=best_model, "bestNumTrees"=best_num_trees, "accuracies"=results)
  return(output)
}

#######

# Set random seed for replicability
set.seed(54321)

# Load data
data(Glass)
data_df <- Glass
y_col_name <- 'Type'

if(check_na_inf_presence(data_df)){
  # If there is na or inf values, remove
  print("NA or INF values detected. Cleaning rows...")
  data_df = na.omit(data_df)
}

data_dims <- dim(data_df)
num_rows <- data_dims[1]
num_features <- data_dims[2]

sprintf("There are %s examples and %s features", data_dims[1], data_dims[2]-1)
levels(data_df[[y_col_name]])

# Look at the structure of data
head(data_df)
str(data_df)
summary(data_df)


# Plot data y-label distribution to observe data imbalance
plot_data_distribution(data_df, y_col_name, "Distribution of Glass Types", "Glass Type", "Frequency", "dataset_glass_type_distr.png")

# Plot feature correlation
plot_feature_correlation_heatmap(data_df, "correlation_heatmap.png")


# Plot Boxplot
plot_boxplot(data_df[, -ncol(data_df)], "Raw Data", "raw_data_distribution.png")
summary(data_df)

# Normalize data
data_df_scaler <- preProcess(data_df, method = c("center","scale"))
data_df_scaled <-predict(data_df_scaler, data_df)
plot_boxplot(data_df_scaled[, -ncol(data_df)], "Normalized Data", "normalized_data.png")
summary(data_df_scaled)

# Train-test split
sample_idx <- createDataPartition(y=data_df_scaled[[y_col_name]], p=0.8, list=FALSE)

#Create Training  and Testing Sets
train_df <- data_df_scaled[sample_idx,]
test_df <- data_df_scaled[-sample_idx,]


# Class distribution of train and test
plot_data_distribution(train_df, y_col_name, "Distribution of Train Data", "Glass Type", "Frequency", "train_data_class_distribution.png")
plot_data_distribution(test_df, y_col_name, "Distribution of Test Data", "Glass Type", "Frequency", "test_data_class_distribution.png")

train_df <- upSample(x = train_df[, -ncol(train_df)],
                     y = train_df[[y_col_name]])
colnames(train_df)[colnames(train_df)== 'Class'] <- y_col_name

# Class distribution train after upsampling
plot_data_distribution(train_df, y_col_name, "Distribution of Train Data", "Glass Type", "Frequency", "train_data_class_distribution_upsampled.png")

# Fit model with 10 Cross-Validation
trainCtrl <- trainControl(method="cv", number=10)

# Train model
# Tune number of trees
tuning_result <- tune_num_trees(100, 1000)
plot_tree_tuning(tuning_result$accuracies, seq(100, 1000, by=100), "tree_tuning.png")

model_rf <- train(Type~., data=train_df, method='rf', metric='Accuracy', ntree=tuning_result$bestNumTrees, trControl=trainCtrl)
#Predict new data with model fitted
predictions <- predict(model_rf, newdata=test_df)

#Shows Confusion Matrix and performance metrics
confusion_matrix <- confusionMatrix(data=predictions, reference =test_df[[y_col_name]])

plot_confusion_matrix(confusion_matrix, "confusion_matrix_feats.png")

# Plot feature importance
var_imp <- varImp(model_rf)
png(filename="feature_importance.png")
plot(var_imp)
dev.off()


