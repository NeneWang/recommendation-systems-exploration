
This is theFinal Project Analysis of a dataset using the techniques learn on class.

## Dataset

The dataset selected is:

- Game Recommendations on Steam

https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?select=games.csv


### Data


- At least six major variables, including:
- 3 or more continuous variables (price, population, age, dimensions, rating, etc.)
- 3 or more categorical variables (species, product type, political party, home state, etc.)

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





### Description

The dataset contains over 41 million cleaned and preprocessed user recommendations (reviews) from a Steam Store - a leading online platform for purchasing and downloading video games, DLC, and other gaming-related content. Additionally, it contains detailed information about games and add-ons.
Content


The dataset consists of three main entities:

- `games.csv` - a table of games (or add-ons) information on ratings, pricing in US dollars $, release date, etc. A piece of extra non-tabular details on games, such as descriptions and tags, is in a metadata file;
- `users.csv` - a table of user profiles' public information: the number of purchased products and reviews published;
- `recommendations.csv` - a table of user reviews: whether the user recommends a product. The table represents a many-many relation between a game entity and a user entity.

The dataset does not contain any personal information about users on a Steam Platform. A preprocessing pipeline anonymized all user IDs. All collected data is accessible to a member of the general public.

Acknowledgements
The dataset was collected from Steam Official Store. All rights on the dataset thumbnail image belong to the Valve Corporation.

Inspiration
Use this dataset to practice building a game recommendation system or performing an Exploratory Data Analysis on products from a Steam Store.



