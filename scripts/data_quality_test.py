import pandas as pd
import pytest


@pytest.fixture(scope="module")
def old_menu_item_df():
    return pd.read_csv("open_refine_Cleaned/Menu-Item.csv")

@pytest.fixture(scope="module")
def menu_item_df():
    return pd.read_csv("python_cleaned/Menu-Item.csv")


@pytest.fixture(scope="module")
def dish_df():
    return pd.read_csv("python_cleaned/Dish.csv")


@pytest.fixture(scope="module")
def menu_df():
    return pd.read_csv("python_cleaned/Menu.csv")


@pytest.fixture(scope="module")
def menu_page_df():
    return pd.read_csv("python_cleaned/Menu-Page.csv")


def test_missing_dish_prices(menu_item_df):
    missing_dish_price = menu_item_df["price"].isnull().sum()
    assert (
        missing_dish_price == 0
    ), f"There are {missing_dish_price} missing dish prices"


def test_missing_created_dates(menu_item_df):
    missing_created_at = menu_item_df["created_at"].isnull().sum()
    assert (
        missing_created_at == 0
    ), f"There are {missing_created_at} missing created at date"


## I don't think we need to test high price, low price, and menu name as they aren't part of our use case.

# # Test function to check for missing dish high prices
# def test_missing_dish_high_prices(dish_df):
#     missing_dish_high_price = dish_df["highest_price"].isnull().sum()
#     assert (
#         missing_dish_high_price == 0
#     ), f"There are {missing_dish_high_price} missing dish high prices"


# # Test function to check for missing dish low prices
# def test_missing_dish_low_prices(dish_df):
#     missing_dish_low_price = dish_df["lowest_price"].isnull().sum()
#     assert (
#         missing_dish_low_price == 0
#     ), f"There are {missing_dish_low_price} missing dish low prices"


# # Test function to check for missing menu names
# def test_missing_menu_name(menu_df):
#     missing_menu_name = menu_df["name"].isnull().sum()
#     assert missing_menu_name == 0, f"There are {missing_menu_name} missing menu names"


# Test function to check if 'created_at' column is in datetime format
def test_created_at_datetime(menu_item_df):
    try:
        pd.to_datetime(menu_item_df['created_at'],format='ISO8601').dt.time
        is_datetime = True
    except ValueError:
        is_datetime = False
    assert (
        is_datetime
    ), f"'created_at' column is of type {menu_item_df['created_at'].dtype}, and couldn't be converted to datetime"


# Test function to check for duplicate names
def test_no_duplicate_names(dish_df):
    duplicate_names = dish_df.groupby(["name"])["name"].count()
    num_duplicate_names = duplicate_names[duplicate_names > 1].count()

    assert (
        num_duplicate_names == 0
    ), f"There are {num_duplicate_names} duplicate dish names"


def test_no_leading_trailing_whitespace(dish_df):
    dirty_dish_names = (
        dish_df["name"].apply(lambda x: isinstance(x, str) and (x.strip() != x)).sum()
    )

    assert (
        dirty_dish_names == 0
    ), f"There are {dirty_dish_names} dish names with leading and trailing whitespace"


def test_name_consistent_format(dish_df):
    inconsistent_format_count = 0

    for name in dish_df["name"]:
        if not isinstance(name, str):
            # check if the data is of type string
            inconsistent_format_count += 1
        elif name != name.title():
            # check if the name is in title case
            inconsistent_format_count += 1

    assert (
        inconsistent_format_count == 0
    ), f"There are {inconsistent_format_count} names with inconsistent format"


def outliers_removed(olddataframe, dataframe, column_name, multiplier=1.5):
    Q1 = olddataframe[column_name].quantile(0.10)
    Q3 = olddataframe[column_name].quantile(0.90)
    IQR = Q3 - Q1

    # Count the number of outliers
    outliers = dataframe[
        (dataframe[column_name] < Q1 - multiplier * IQR)
        | (dataframe[column_name] > Q3 + multiplier * IQR)
    ]
    outliers_count = outliers.shape[0]

    assert outliers_count == 0, f"There are {outliers_count} outliers in {column_name}"


def test_menu_item_price_outliers(old_menu_item_df, menu_item_df):
    outliers_removed(old_menu_item_df, menu_item_df, "price", multiplier=2.0)

## I don't think we need to test for lowest price and highest price as that isn't part of our analysis
# def test_dish_highest_price_outliers(dish_df):
#     test_outliers_removed(dish_df, "highest_price", multiplier=2.0)


# def test_dish_lowest_price_outliers(dish_df):
#     test_outliers_removed(dish_df, "lowest_price", multiplier=2.0)