import pandas as pd

dish_df = pd.read_csv("open_refine_cleaned/Dish.csv")
menu_item_df = pd.read_csv("open_refine_cleaned/Menu-Item.csv")


## Remove empty rows for prices
menu_item_df.dropna(subset=['price'], inplace=True)
menu_item_df.dropna(subset=['created_at'], inplace=True)

## Dish name to title case
dish_df["name"] = dish_df["name"].str.title()

## Merge duplicate dish names
# 1. Get the ids of duplicate names in dish_df.
# 2. Switch all dish_ids in menu_page to the first id
# 3. Remove all but the first from dish_df.
ids = dish_df["name"]
duped_name = dish_df[ids.isin(ids[ids.duplicated()])].sort_values("name")
duped_ids = duped_name.groupby(['name'])['id'].apply(lambda x: ','.join([str(y) for y in x])).reset_index()
ids_to_drop = []

for row in duped_ids.iterrows():
    dish_ids = row[1]["id"].split(",")
    first = int(dish_ids[0])
    for id in dish_ids[1:]:
        ids_to_drop.append(int(id))
        menu_item_df.loc[menu_item_df["dish_id"] == int(id), "dish_id"] = first
        
dish_df = dish_df[~dish_df['id'].isin(ids_to_drop)]

## Remove outlier prices
q_low = menu_item_df["price"].quantile(0.10)
q_hi  = menu_item_df["price"].quantile(0.90)
iqr = q_hi - q_low
mul = 2.0

menu_item_df = menu_item_df[(menu_item_df["price"] < q_hi + mul * iqr) & (menu_item_df["price"] > q_low - mul * iqr)]

## Normalize prices
menu_item_df["price"] = (menu_item_df["price"] - menu_item_df["price"].min()) / (menu_item_df["price"].max() - menu_item_df["price"].min())

## Remove dishes without menu items and menu items without dishes
dish_df = dish_df[dish_df['id'].isin(menu_item_df["dish_id"])]
menu_item_df = menu_item_df[menu_item_df['dish_id'].isin(dish_df["id"])]


## Write cleaned data to files
dish_df.to_csv("python_cleaned/Dish.csv", index=False)
menu_item_df.to_csv("python_cleaned/Menu-Item.csv", index=False)