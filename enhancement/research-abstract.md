## Exploration Abstract

### Data Cleaning

- Make sure that there is at least 5 count of books:

books_ratings_count_df_more_than_five = books_ratings_count_df[books_ratings_count_df['count'] > 5]

Filter raters where had less than 5 ratings performed.


# Conclusions

- Have two testing Algorithms:
  - One with all users
  - One with users with no more than 200 ratings
  - One with users with no more than 500 ratings
- Removals of low rated books:
  - Removal of books with only a singular rating?
  - Removal of books with less than total 5 ratings

## Insights

- The highest raters users tend to rate books lower.
- The 58.86 % of books have less than 2 ratings. (32471 books left)

## To be done:

- [ ] Split Datasets.






