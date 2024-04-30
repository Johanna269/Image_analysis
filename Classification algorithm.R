#Get all pictures
im.names <- list.files(path ="//Users/johannaschoellhorn/Desktop/AUD/Exam/Face data for students /All Faces/", pattern = "\\.png$")
im.list = list()

#Extracting labels dataframe
file_names <- list.files(path ="//Users/johannaschoellhorn/Desktop/AUD/Exam/Face data for students /All Faces/", pattern = "\\.png$")
set.seed(123)
selected_file_names <- sample(file_names, 200)
file_df_all <- data.frame(filename = selected_file_names)

#Extract pictures
set.seed(123)
selected.im.names <- sample(im.names, 200)

#Load the pictures
for (i in 1:length(selected.im.names)) {
  im.list[[i]] <- load.image(paste("//Users/johannaschoellhorn/Desktop/AUD/Exam/Face data for students /All Faces/", selected.im.names[i], sep=""))
}

# Add pictures into one list & create labels dataframe
all_pictures <- c(im.list, im.list_target)
images <- rbind(file_df, file_df_all)

# Create a binary identifier 
identifier_im_list_target <- all_pictures %in% im.list #Will evaluate to TRUE if it is the target emotion

#add to files dataframe
images <- cbind(images, identifier_im_list_target)
images <- rename(images, identifier = identifier_im_list_target)


#Eigenfaces and PCA
picture_matrix <- do.call(rbind, all_pictures)

# Applying PCA
images_simplified <- prcomp(picture_matrix, center = TRUE)
cumulative_variance <- summary(images_simplified)$importance[3,]
max_eigenvector <- which.max(cumulative_variance >= 0.85)
max_eigenvector #46 principal components are sufficient to explain 85% of the variance
eigenface_scores <- images_simplified$x[, 1:max_eigenvector]

# Retain eigenface loadings
labels <- names(unlist(all_pictures))
eigenface_scores_with_labels <- cbind(eigenface_scores, label = labels)
eigenfaces <- as.data.frame(eigenface_scores_with_labels)

#Add to files dataframe
images <- cbind(images, eigenfaces)



#Gabor filter

#Ensure all pictures are grayscale
img_gray <- lapply(all_pictures, grayscale)
for (i in 1:length(img_gray)){
  img_gray[[i]] <- img_gray[[i]][, , 1,1]
}

gabor_filtered <- list()

#Prepare metrics
total_iterations <- length(seq(from = 4, to = 8, by = 1)) * 
  length(seq(from = 60, to = 150, by = 30)) * 
  length(seq(from = 1, to = 3, by = 0.5))
num_pictures <- length(img_gray)
mean_gabor_filter <- matrix(nrow = num_pictures, ncol = total_iterations)
sd_gabor_filter <- matrix(nrow = num_pictures, ncol = total_iterations)

# Loop through images
for (i in 1:num_pictures) {
  y <- 1
  for (lambda in seq(from = 4, to = 8, by = 1)) {
    for (theta in seq(from = 60, to = 150, by = 30)) {
      for (bw in seq(from = 1, to = 3, by = 0.5)) {
        # Apply Gabor filter
        gabor_filtered[[i]] <- gabor.filter(img_gray[[i]], lamda = lambda, theta = theta, bw = bw)
        # Calculate the mean of attribute "filtered_img"
        mean_gabor_filter[i, y] <- mean(gabor_filtered[[i]][["filtered_img"]])
        # Calculate the SD of attribute "filtered_img"
        sd_gabor_filter[i, y] <- sd(gabor_filtered[[i]][["filtered_img"]])
        y <- y + 1
      }
    }
  }
}


#Convert the mean and SD matrices to data frames
mean_gabor_df <- as.data.frame(mean_gabor_filter)
names(mean_gabor_df) <- paste("mean_gabor", 1:ncol(mean_gabor_df), sep = "_")
sd_gabor_df <- as.data.frame(sd_gabor_filter)
names(sd_gabor_df) <- paste("sd_gabor", 1:ncol(sd_gabor_df), sep = "_")


#Add to dataframe
images <- cbind(images, mean_gabor_df, sd_gabor_df)


#Plotting Gabor filter

#Rotate the image
rotate_clockwise <- function(x) {
  t(apply(x, 2, rev))
}

rotated_img <- rotate_clockwise(t(gabor_filtered[[206]][["filtered_img"]]))

#Plotting
image(rotated_img, col = gray((0:58)/58), main = "Gabor filter example", useRaster = TRUE, asp = 1, axes = FALSE)


#Hough lines

hl=list() #Initiate list 

#Applying hough line function on top of edge detection function
for (i in 1:length(all_pictures)) {
  void <- hough_line(cannyEdges(as.cimg(all_pictures[[i]])), ntheta = 800, data = TRUE)
  hl[[i]] <- subset(void,score > quantile(score,.9995)) #Keep only strong lines (above 99.95% quantile)
}

#Add to dataframe
images$n_lines <- unlist(lapply(hl,FUN=width))

# Plotting hough lines
plot(all_pictures[[206]]); nfline(theta = hl[[1]]$theta, rho = hl[[1]]$rho, col = "red")



#Training a classification model, using eigenfaces, gabor filter, and hough lines
set.seed(123)
training <- sample(1:nrow(images), 0.8*nrow(images))
train_data <- images[training,]
test_data <- images[-training,]

#Create target to train on 
target_train <- as.factor(train_data$identifier)
target_test <- as.factor(test_data$identifier)


#Extract eigenface features only from dataset
set.seed(123)
train_eigenface <- train_data[ , -c(1,2,49:249)]
test_eigenface <- test_data[ , -c(1,2,49:249)]

#Extract n_lines and mean_gabor features only from dataset
train_features <- train_data[ , -c(1,2,3:48)]
test_features <- test_data[ , -c(1,2,3:48)]

#Extract all features from dataset
train_combined <- train_data[ , -c(1, 2)]
test_combined <- test_data[ , -c(1, 2)]

#SVM
#Create svm model 
svmmodel_eigenface <- svm(target_train ~ ., data = train_eigenface, type = "C-classification")
svmmodel_features <- svm(target_train ~ ., data = train_features, type = "C-classification")
svmmodel_combined <- svm(target_train ~ ., data = train_combined, type = "C-classification")

#Make predictions with svm model: TEST
eigenface_svm_pred_test <- predict(svmmodel_eigenface, test_eigenface)
features_svm_pred_test <- predict(svmmodel_features, test_features)
combined_svm_pred_test <- predict(svmmodel_combined, test_combined)

#Calculate recall:TEST
svm_eigenface_recall_test <- Recall(target_test, eigenface_svm_pred_test)
svm_features_recall_test <- Recall(target_test, features_svm_pred_test)
svm_combined_recall_test <- Recall(target_test, combined_svm_pred_test)

# F1 score:TEST
svm_eigenface_f1score_test <- F1_Score(target_test, eigenface_svm_pred_test)
svm_features_f1score_test <- F1_Score(target_test, features_svm_pred_test)
svm_combined_f1score_test <- F1_Score(target_test, combined_svm_pred_test)

#Calculte accuracy: TEST
svm_eigenface_accuracy_test <- Accuracy(target_test, eigenface_svm_pred_test)
svm_features_accuracy_test <- Accuracy(target_test, features_svm_pred_test)
svm_combined_accuracy_test <- Accuracy(target_test, combined_svm_pred_test)

#Make comprehensive tables
cat(sprintf("Eigenface Evaluation (SVM Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            svm_eigenface_accuracy_test, svm_eigenface_recall_test, svm_eigenface_f1score_test))
cat(sprintf("Features Evaluation (SVM Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            svm_features_accuracy_test, svm_features_recall_test, svm_features_f1score_test))
cat(sprintf("Combined Evaluation (SVM Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            svm_combined_accuracy_test, svm_combined_recall_test, svm_combined_f1score_test))


#Random Forest
#Build RF model
rfmodel_eigenface <- randomForest(target_train ~ ., data = train_eigenface)
rfmodel_features <- randomForest(target_train ~ ., data = train_features)
rfmodel_combined <- randomForest(target_train ~ ., data = train_combined)

#Make predictions with random forest model: TEST
rf_eigenface_pred_test <- predict(rfmodel_eigenface, test_eigenface)
rf_features_pred_test <- predict(rfmodel_features, test_features)
rf_combined_pred_test <- predict(rfmodel_combined, test_combined)

#Calculate recall random forest: TEST
rf_eigenface_recall_test <- Recall(target_test, rf_eigenface_pred_test)
rf_features_recall_test <- Recall(target_test, rf_features_pred_test)
rf_combined_recall_test <- Recall(target_test, rf_combined_pred_test)

#Calculate F1 score random forest: TEST
rf_eigenface_f1score_test <- F1_Score(target_test, rf_eigenface_pred_test)
rf_features_f1score_test <- F1_Score(target_test, rf_features_pred_test)
rf_combined_f1score_test <- F1_Score(target_test, rf_combined_pred_test)

#Calculate accuracy random forest: TEST
rf_eigenface_accuracy_test <- Accuracy(target_test, rf_eigenface_pred_test)
rf_features_accuracy_test <- Accuracy(target_test, rf_features_pred_test)
rf_combined_accuracy_test <- Accuracy(target_test, rf_combined_pred_test)

#Make comprehensive tables
cat(sprintf("Eigenface Evaluation (Random Forest Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            rf_eigenface_accuracy_test, rf_eigenface_recall_test, rf_eigenface_f1score_test))
cat(sprintf("Features Evaluation (Random Forest Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            rf_features_accuracy_test, rf_features_recall_test, rf_features_f1score_test))
cat(sprintf("Combined Evaluation (Random Forest Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            rf_combined_accuracy_test, rf_combined_recall_test, rf_combined_f1score_test))

#Naive Bayes
#Build NB model
nbmodel_eigenface <- naiveBayes(target_train ~ ., data = train_eigenface)
nbmodel_features <- naiveBayes(target_train ~ ., data = train_features)
nbmodel_combined <- naiveBayes(target_train ~ ., data = train_combined)

#Make predictions with NB model: TEST
nb_eigenface_pred_test <- predict(nbmodel_eigenface, test_eigenface)
nb_features_pred_test <- predict(nbmodel_features, test_features)
nb_combined_pred_test <- predict(nbmodel_combined, test_combined)

#Calculate recall NB: TEST
nb_eigenface_recall_test <- Recall(target_test, nb_eigenface_pred_test)
nb_features_recall_test <- Recall(target_test, nb_features_pred_test)
nb_combined_recall_test <- Recall(target_test, nb_combined_pred_test)

#Calculate F1 score NB: TEST
nb_eigenface_f1score_test <- F1_Score(target_test, nb_eigenface_pred_test)
nb_features_f1score_test <- F1_Score(target_test, nb_features_pred_test)
nb_combined_f1score_test <- F1_Score(target_test, nb_combined_pred_test)

#Calculate accuracy NB: TEST
nb_eigenface_accuracy_test <- sum(nb_eigenface_pred_test == target_test) / length(target_test)
nb_features_accuracy_test <- sum(nb_features_pred_test == target_test) / length(target_test)
nb_combined_accuracy_test <- sum(nb_combined_pred_test == target_test) / length(target_test)

#Comprehensive table
cat(sprintf("Eigenface Evaluation (Naive Bayes Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            nb_eigenface_accuracy_test, nb_eigenface_recall_test, nb_eigenface_f1score_test))
cat(sprintf("Features Evaluation (Naive Bayes Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            nb_features_accuracy_test, nb_features_recall_test, nb_features_f1score_test))
cat(sprintf("Combined Evaluation (Naive Bayes Testing data): \nAccuracy: %f\nRecall: %f\nF1 Score: %f\n\n",
            nb_combined_accuracy_test, nb_combined_recall_test, nb_combined_f1score_test))
