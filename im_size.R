# ConvTranspose2d
x <- 1
layers <- c(4,1,0,
            4,2,1,
            4,2,1,
            4,2,3)
layers <- matrix(layers, ncol = 3, byrow = TRUE)
for (i in 1:nrow(layers)) {
  k <- layers[i,1]
  s <- layers[i,2]
  p <- layers[i,3]
  x <- (x - 1) * s - 2*p + k
  print(x)
}




# Conv2d
x <- 32
layers <- c(4,1,0,
            4,2,1,
            4,2,1,
            4,4,2)
layers <- matrix(layers, ncol = 3, byrow = TRUE)
for (i in 1:nrow(layers)) {
  k <- layers[i,1]
  s <- layers[i,2]
  p <- layers[i,3]
  x <- ceiling((x + 2 * p - k) / s) + 1
  print(x)
}