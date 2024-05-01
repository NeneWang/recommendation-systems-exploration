## Report for Check in 1


[The Project Repository](https://github.com/NeneWang/recommendation-systems-exploration/tree/master/final_project_exploration) 

Please submit a link to the dataset you plan to use, and at least 2 high-level analyses (Section 3 of the project writeup). For each analysis, please submit at minimum:

The question you plan to answer
A visualization you plan to create, if applicable
A statistical test you plan to use, if applicable
You do not have to have performed the analyses yet.

I will leave feedback on your suggested analyses and if any other analyses might be appropriate for your dataset.


## Question and Plan


Given the following dataset: available at https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?select=recommendations.csv


### Data

games.csv

| Feature        | Variable Type | Description                 | Example            | Category   |
| -------------- | ------------- | --------------------------- | ------------------ | ---------- |
| app_id         | int           | Unique identifier of a game | 113020             | Identifier |
| title          | string        | Title of a game             | Escape Dead Island | Identifier |
| date_release   | date          | Release date of a game      | 2014-11-21         | Continuous |
| win            | bool          | Windows OS support          | True               | Category   |
| mac            | bool          | Mac OS support              | False              | Category   |
| linux          | bool          | Linux OS support            | False              | Category   |
| rating         | float         | Average rating of a game    | 3.0                | Continuous |
| positive ratio | float         | Ratio of positive reviews   | 0.5                | Continuous |
| user_reviews   | int           | Number of user reviews      | 0                  | Continuous |


recommendations.csv

| feature        | Variable Type | Description                      | Example    | Category   |
| -------------- | ------------- | -------------------------------- | ---------- | ---------- |
| app_id         | int           | Unique identifier of a game      | 113020     | Identifier |
| helpful        | int           | Number of helpful reviews        | 0          | Continuous |
| funny          | int           | Number of funny reviews          | 0          | Continuous |
| date           | date          | Date of a review                 | 2014-11-21 | Continuous |
| is_recommended | bool          | Whether a user recommends a game | True       | Category   |
| hours          | float         | Number of hours played           | 0.0        | Continuous |
| user_id        | int           | Unique identifier of a user      | 5250       | Identifier |
| review_id      | int           | Unique identifier of a review    | 1          | Identifier |

users.csv

| Feature  | Variable Type | Description                 | Example | Category   |
| -------- | ------------- | --------------------------- | ------- | ---------- |
| user_id  | int           | Unique identifier of a user | 5250    | Identifier |
| products | int           | Count of purchased products | 1       | Continuous |
| reviews  | int           | Count of published reviews  | 0       | Continuous |

## *Questions I want to answer

- Find if there is a correlation between the number of reviews and the number of products purchased by a user.
- Does players with more hours played provide more helpful reviews? What is the relation between hours played and the review result?
- Find if there is an relation with the users count of games and total hours played and the reviews posted. 
- Find if there are relations between: rating, prices, reviews, positive ratio ,ratio
- Find the clusters of users based on reviews provided and activity.
- Find the clusters of games based on the reviews


### Plan: Find the clusters of users based on reviews provided and activity.
Given the following View of the Data Users focused data find the clusters and try to find the clusters

e.g. 
A join table of user, products, totla_hours_played, average_recommendation_rating and total_reviews

- What does the cluster look of users look like based on the joined table of users with total hours p

## Visualization Planned

- Clustering from the perspective of:
  - Games
  - Recommendation
  - Users
- Relationship if any between:
  - Number of reviews and the number of products purchased by a user
  - Hours played and helpful reviews
  - Users count of games and total hours played and the reviews posted
  - Rating, prices, reviews, positive ratio, ratio

## Statistical Tests

- If any relationships are found between the variables, I will use a correlation test to determine the strength of the relationship.
- If clusters are found I will try to use some intuition to find the relationship and run the appropriate statistical tests to understand the clusters.


