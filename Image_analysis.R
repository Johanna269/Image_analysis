#Preperation
library(dplyr)
library(ggplot2)
library(pixmap)

image_names <- list.files(path = "/Users/johannaschoellhorn/Desktop/Master/Analysing U.D/Assignments/Assignment 3/faces/", pattern = NULL, all.files = FALSE, full.names = FALSE)

image_names_complete <- paste(path = "/Users/johannaschoellhorn/Desktop/Master/Analysing U.D/Assignments/Assignment 3/faces/", image_names[1:1000], sep = "")

#Store the matrices in a list called images and plotting an image
images <- lapply(image_names_complete, function(path) read.pnm(path))
plot(images[[944]])


#Extracting the gray scale values for each image.
image_total_aux <- lapply(images, function(x) as.vector(t(x@grey)))
image_total <- do.call(rbind, image_total_aux)


#Extracting an example image
example_image <- matrix(image_total[944, ], nrow = 64, ncol = 64)
image(example_image[, ncol(example_image):1], col = grey((0:255)/255), axes = FALSE, main = "Example: Arnold Schwarzenegger")


#Creating bins to show the distribution of the gray scale
hist(image_total[944,], xlim = c(0, 1), xlab = 'grey value', main = 'Grey value in Arni picture')
 

#Computing the average face
mean_face <- colMeans(image_total)
mean_face_matrix <- matrix(mean_face, nrow = 64, ncol = 64)
image(mean_face_matrix[, ncol(mean_face_matrix):1], col = grey((0:255)/255), axes = FALSE, main = "Mean face")


#Principal Components
image_simplefied <- prcomp(image_total)

cumulative_variance <- summary(image_simplefied)$importance[3,]
max_eigenvector <- which.max(cumulative_variance >= 0.9)
print(max_eigenvector)
.

#Plotting eigenfaces
par(mfrow = c(2, 5)) 
for (i in 1:10) {
  face_matrix <- matrix(image_simplefied$rotation[,i], nrow = 64, ncol = 64)
  image(face_matrix[, ncol(face_matrix):1], col = grey((0:255)/255), axes = FALSE, main = i)
}
```

#Recovering a face
n <- max_eigenvector

recovered_matrix <- image_simplefied$x[, 1:n] %*% t(image_simplefied$rotation[, 1:n])
recovered_matrix <- scale(recovered_matrix, center = -image_simplefied$center, scale = FALSE)

image_index <- 944 
# Orig. image
original_image <- matrix(image_total[image_index, ], nrow = 64, ncol = 64, byrow = TRUE)
# Recov. image
recovered_image <- matrix(recovered_matrix[image_index, ], nrow = 64, ncol = 64, byrow = TRUE)

#Plot comparison
par(mfrow = c(1, 2)) 
image(1:64, 1:64, t(original_image)[, ncol(t(original_image)):1], col = gray((0:255)/255), 
      main = "Original Face", xlab = "", ylab = "", axes = FALSE)

image(1:64, 1:64, t(recovered_image)[, ncol(t(recovered_image)):1], col = gray((0:255)/255), 
      main = "Recovered Face", xlab = "", ylab = "", axes = FALSE)


#Image detection
# Focal image
focal_image_index <- 944
focal_image <- image_total[focal_image_index, ] - image_simplefied$center

focal_image_matrix <- matrix(focal_image, nrow = 1)

focal_image_projection <- focal_image_matrix %*% image_simplefied$rotation[, 1:max_eigenvector]

# Calculate Euclidean distances
distances <- apply(image_simplefied$x[, 1:max_eigenvector], 1, function(projection) sum((projection - focal_image_projection)^2))

# Index with the minimum distance
min_distance_index <- which.min(distances)
min_distance_index
