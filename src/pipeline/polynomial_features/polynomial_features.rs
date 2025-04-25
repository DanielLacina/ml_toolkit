use crate::dataframe::{DataFrame, DataType, DataTypeValue};  
use crate::pipeline::transformers::Transformer; 

pub struct PolynomialFeatures {
    degrees: u32
}

impl PolynomialFeatures {
    pub fn new(degrees: u32) -> Self {
        Self {
            degrees
        }
    } 
}

impl Transformer for PolynomialFeatures {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df_with_polynomial_features = df.clone(); 
        for column_name in column_names {
            let (dtype, values) = df.get_column(column_name); 
            if !matches!(dtype, DataType::Float) {
                panic!("column {} must have a datatype of float", column_name);
            }
            if self.degrees > 1 {
                for i in (2..self.degrees + 1) {
                    let scaled_values: Vec<DataTypeValue> = values.iter().map(|value| {
                        let value = match value {
                            DataTypeValue::Float(inner) => {
                                inner
                            }, _ => {
                                panic!("{:?} is inconsistent with column {} datatype of {:?}", value, column_name, dtype);
                            }
                        }; 
                        DataTypeValue::Float(value.powf(i as f32))
                    }
                    ).collect();   
                    let scaled_column_name =  format!("{}^{}", column_name, i); 
                    df_with_polynomial_features.insert_column(&scaled_column_name,  &scaled_values, dtype);
                } 
            }
        } 
        return df_with_polynomial_features;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv; 
    use std::iter::zip;

    #[test]
    fn test_polynomial_features() {
        let degrees = 3;
        let df = df_from_csv("housing.csv", Some(100)); 
        let numeric_columns = df.numeric_columns(); 
        let numeric_columns: Vec<String> = numeric_columns.into_iter().map(|column| column.clone()).collect();
        let polynomial_features = PolynomialFeatures::new(degrees);
        let df_with_polynomial_features = polynomial_features.transform(&df, &numeric_columns); 
        assert!(numeric_columns.into_iter().all(|column| {
             let (_, values) = df_with_polynomial_features.get_column(&column); 
             for i in (2..degrees + 1) {
                let scaled_column_name = format!("{}^{}", column, i);
                let (_, scaled_values) = df_with_polynomial_features.get_column(&scaled_column_name);
                if !(zip(scaled_values, values).all(|(scaled_value, value)| {
                    let value = match value {
                        DataTypeValue::Float(inner) => {
                           inner 
                        },
                        _ => panic!("value must be float type")
                    };
                    DataTypeValue::Float(value.powf(i as f32)) == *scaled_value  
                })) {
                    return false;
                }
             } 
             return true;
        }));
    }
}

