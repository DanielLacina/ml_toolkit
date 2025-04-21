use crate::dataframe::{DataFrame, DataType, DataTypeValue};
use std::collections::HashMap;



pub fn df_with_one_hot_encoding(df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
    let columns_of_new_df: Vec<String> = df.columns().into_iter().filter(|col_name| !column_names.contains(col_name) ).map(|col_name| col_name.clone()).collect();     
    let mut df_with_one_hot_encoding = df.get_columns_as_df(&columns_of_new_df); 
    for column_name in column_names {
        let (_, values) = df.get_column(column_name);
        let categorical_values = extract_categorical_values(values);
        let mut categories = categorical_values.iter().map(|(category, _)| category.clone()).collect::<Vec<String>>();  
        categories.sort();
        for category in categories.iter() {
            let cat_values = categorical_values.get(category).unwrap();
            df_with_one_hot_encoding.insert_column(category, cat_values, &DataType::Float);         
        }
    }
    return df_with_one_hot_encoding;
}

  fn extract_categorical_values(values: &Vec<DataTypeValue>) -> HashMap<String, Vec<DataTypeValue>> {
        let mut categories: HashMap<String, Vec<DataTypeValue>> = HashMap::new();
        for (i, value) in values.iter().enumerate() {
            let inner = match value {
                DataTypeValue::String(inner) => inner.clone(),
                DataTypeValue::Null => "null".to_string(),
                _ => {
                    panic!("dtype value is not categorical")
                }
            };
            for (category, cat_values) in categories.iter_mut() {
                if *category == *inner {
                    cat_values.push(DataTypeValue::Float(1.0));
                } else {
                    cat_values.push(DataTypeValue::Float(0.0));
                }
            }
            if !categories.contains_key(inner.as_str()) {
                let mut cat_values = vec![DataTypeValue::Float(0.0); i];
                cat_values.push(DataTypeValue::Float(1.0));
                categories.insert(inner.clone(), cat_values);
            }
        }
        return categories;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv;  

    #[test]
    fn test_df_with_one_hot_encoding() {
        let filename = "housing.csv";
        let row_limit = 10000;
        let df = df_from_csv(filename, Some(row_limit));
        let categorical_columns= vec!["ocean_proximity".to_string()]; 
        let df_with_one_hot_encoding = df_with_one_hot_encoding(&df, &categorical_columns); 
        println!("{:?}", df_with_one_hot_encoding.columns());
    }
}