
## Setup


Get the following data and save it in a `data` folder on root.:

https://www.kaggle.com/datasets/saurabhbagchi/books-dataset/data

Should have the following files in `data folder`:



```
02/28/2024  12:20 PM    <DIR>          .
03/07/2024  06:17 PM    <DIR>          ..
02/28/2024  12:20 PM        73,293,635 Books.csv
02/28/2024  12:20 PM           318,713 DeepRec.png
02/28/2024  12:20 PM        22,633,892 Ratings.csv
02/28/2024  12:20 PM        11,017,438 Users.csv
```



```
jupyter notebook .
```

| Metric              | Description                                                                                                                       | Formula                                                                                                                                          | Interpretation                      |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| Mean Absolute Error | Measures the average absolute difference between the true values and the predicted values.                                        | $(\text{MAE} = \frac{1}{n} \sum_{i=1}^{n})$                                                                                                      |                                     | Lower is better. |
| Mean Squared Error  | Measures the average of the squares of the differences between the true values and the predicted values.                          | $( \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2 )$                                                        | Lower is better.                    |
| R² Score            | Represents the proportion of the variance in the dependent variable (y) that is predictable from the independent variable(s) (x). | $( R^2 = 1 - \frac{\sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2}{\sum_{i=1}^{n} (y_{\text{true}, i} - \bar{y}_{\text{true}})^2} )$ | Closer to 1 indicates a better fit. |
| Precision           | Measures the proportion of true positive predictions among all positive predictions made.                                         | $( \text{Precision} = \frac{TP}{TP + FP} )$                                                                                                      | Higher is better.                   |
| Recall              | Measures the proportion of true positive predictions among all actual positive instances.                                         | $( \text{Recall} = \frac{TP}{TP + FN} )$                                                                                                         | Higher is better.                   |

This table focuses solely on the description, formula, and interpretation of each metric.