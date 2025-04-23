use crate::{dataframe::{DataFrame, DataType, DataTypeValue}, pipeline::transformers::Transformer};
use std::collections::HashMap;

pub struct OneHotEncoder {
    drop: bool 
}

impl OneHotEncoder {
    pub fn new(drop: bool) -> Self {
        Self {
            drop
        }
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

    //     #[test]
    //     fn test_df_one_hot_encoded() {
    //         let filename = "housing.csv";
    //         let row_limit = 10000;
    //         let df = df_from_csv(filename, Some(row_limit));
    //         let categorical_columns = vec!["ocean_proximity".to_string()];
    //         let one_hot_encoder = OneHotEncoder::new();
    //         let df_one_hot_encoded = one_hot_encoder.df_one_hot_encoded(&df, &categorical_columns);
    //         println!("{:?}", df_one_hot_encoded.columns());
    //     }
}
