use crate::{
    dataframe::{DataFrame, DataType, DataTypeValue},
    pipeline::transformers::Transformer,
};
use std::collections::HashMap;

pub struct OneHotEncoder {
    drop: bool,
}

impl OneHotEncoder {
    pub fn new(drop: bool) -> Self {
        Self { drop }
    }
    fn extract_categorical_values(
        &self,
        column_name: &str,
        values: &Vec<DataTypeValue>,
    ) -> HashMap<String, Vec<DataTypeValue>> {
        let mut categories: HashMap<String, Vec<DataTypeValue>> = HashMap::new();
        let null_placeholder = "null".to_string();
        for (i, value) in values.iter().enumerate() {
            let inner = match value {
                DataTypeValue::String(inner) => inner.clone(),
                DataTypeValue::Null => null_placeholder.clone(),
                _ => {
                    panic!("dtype value is not categorical")
                }
            };
            assert!(
                categories
                    .iter()
                    .all(|(category, _)| *category != null_placeholder),
                "{}",
                format!(
                    "Column {} has designated category of {}",
                    column_name, null_placeholder
                )
            );
            for (category, cat_values) in categories.iter_mut() {
                if *category == *inner {
                    cat_values.push(DataTypeValue::Float(1.0));
                } else {
                    cat_values.push(DataTypeValue::Float(0.0));
                }
            }
            if *inner != null_placeholder {
                if !categories.contains_key(inner.as_str()) {
                    let mut cat_values = vec![DataTypeValue::Float(0.0); i];
                    cat_values.push(DataTypeValue::Float(1.0));
                    categories.insert(inner.clone(), cat_values);
                }
            }
        }
        return categories;
    }
}
impl Transformer for OneHotEncoder {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df_one_hot_encoded = df.clone();
        for column_name in column_names {
            let (_, values) = df.get_column(column_name);
            let categorical_values = self.extract_categorical_values(column_name.as_str(), values);
            let mut categories = categorical_values
                .iter()
                .map(|(category, _)| category.clone())
                .collect::<Vec<String>>();
            categories.sort();
            if self.drop {
                let cat_len = categories.len();
                categories.remove(cat_len - 1);
            }
            for category in categories.iter() {
                let cat_values = categorical_values.get(category).unwrap();
                df_one_hot_encoded.insert_column(category, cat_values, &DataType::Float);
            }
        }
        for column_name in column_names {
            df_one_hot_encoded.remove_column(column_name);
        }
        return df_one_hot_encoded;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv;

    fn test_df_one_hot_encoded(drop: bool) {
        let filename = "housing.csv";
        let row_limit = 1000;
        let df = df_from_csv(filename, Some(row_limit));
        let categorical_column = "ocean_proximity";
        let one_hot_encoder = OneHotEncoder::new(false);
        let df_one_hot_encoded =
            one_hot_encoder.transform(&df, &vec![categorical_column.to_string()]);
        let df_one_hot_encoded_columns = df_one_hot_encoded.columns();
        let categories = df.get_value_frequencies(categorical_column);
        let mut categories: Vec<String> = categories
            .into_iter()
            .map(|(value, _)| match value {
                DataTypeValue::String(inner) => inner,
                _ => panic!("datatype must be a string"),
            })
            .collect();
        if drop {
            categories.sort();
            categories.remove(categories.len() - 1);
        }
        assert!(
            categories
                .iter()
                .all(|category| { df_one_hot_encoded_columns.contains(&category) })
        );
        let category_columns: Vec<(&DataType, &Vec<DataTypeValue>)> = categories
            .iter()
            .map(|category| df_one_hot_encoded.get_column(&category))
            .collect();
        let mut one_count = 0;
        assert!((0..df.len()).all(|i| {
            for (_, values) in category_columns.iter() {
                let mut found_one = false;
                let value = values.get(i).unwrap();
                if *value == DataTypeValue::Float(1.0) {
                    if found_one {
                        return false;
                    } else {
                        found_one = true;
                        one_count += 1;
                    }
                } else if *value == DataTypeValue::Float(0.0) {
                } else {
                    return false;
                }
            }
            return true;
        }));
        assert!(one_count > 0);
    }

    #[test]
    fn test_df_one_hot_encoded_no_drop() {
        test_df_one_hot_encoded(false);
    }

    #[test]
    fn test_df_one_hot_encoded_drop() {
        test_df_one_hot_encoded(true);
    }

    #[test]
    #[should_panic]
    fn test_df_one_hot_encoded_with_numeric_column() {
        let filename = "housing.csv";
        let row_limit = 1000;
        let df = df_from_csv(filename, Some(row_limit));
        let numeric_column = "median_income";
        let one_hot_encoder = OneHotEncoder::new(false);
        one_hot_encoder.transform(&df, &vec![numeric_column.to_string()]);
    }
}
