### Churn Prediction Project

library(tidyverse)
library(dplyr)
library(caret)
library(MLmetrics)

### Import dataset

df <- read.csv("sales-churn-project.csv")

### Check Data

glimpse(df)
View(df)
mean(complete.cases(df))

# convert 'churn' from 'chr' to 'fct'
df$churn <- as.factor(df$churn)

### Check for Imbalance class
check_imbalance <- df %>%
  count(churn) %>%
  mutate(pct = n/sum(n))

print(check_imbalance)

#####################################################################

### 1. Split Data
set.seed(99)
id <- createDataPartition(y = df$churn,
                          p = 0.8,
                          list = FALSE)
train_df <- df[id, ]
test_df <- df[-id, ]

#####################################################################

### 2. Train Model

# churn -> yes, no -> binary classification problems
# 1. Focus on logistic regression first
# 2. Using 'ROC' due to Imbalance class issue

# Logistic regression  
set.seed(99)

  # set Train control
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = prSummary,
    verboseIter = TRUE
  )

logistic_model <- train(churn ~ .,
                        data = train_df,
                        method = "glm",
                        metric = "ROC",
                        preProcess = c("center", "scale", "nzv"),
                        trControl = ctrl
                        )
print(logistic_model)

######################################################################

# Random forest
set.seed(99)

myGrid <- data.frame(mtry = 2:7)

rf_model <- train(churn ~ .,
                  data = train_df,
                  method = "rf",
                  metric = "AUC",
                  preProcess = c("center", "scale", "nzv"),
                  tuneGrid = myGrid,
                  trControl = ctrl
                  )
print(rf_model)

######################################################################

# Compare Model's metric

compare_model_results <- resamples(list(
  logistic = logistic_model,
  randomForest = rf_model
  ))

summary_compare_model <- summary(compare_model_results)

print(summary_compare_model)

### Random forest model showed better performance in all metrics.


######################################################################

### 3. Test Model

p_rf <- predict(rf_model,
                newdata = test_df)

result_rf <- confusionMatrix(p_rf, test_df$churn,
                mode = "prec_recall",
                positive = "Yes")

print(result_rf)