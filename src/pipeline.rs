use crate::dataframe::{DataFrame, DataType, DataTypeValue};
use std::collections::HashMap;

pub enum ImputerStrategy {
    Median,
}

pub enum StringEncoding {
    OneHot,
}

pub enum Scalar {
    None,
    Standard,
}

pub struct Pipeline {
    string_encoding: StringEncoding,
    imputer_strategy: ImputerStrategy,
    scalar: Scalar,
}

impl Pipeline {
    pub fn new(
        string_encoding: StringEncoding,
        imputer_strategy: ImputerStrategy,
        scalar: Scalar,
    ) -> Self {
        Self {
            string_encoding,
            imputer_strategy,
            scalar,
        }
    }

    pub fn df_to_matrix(&self, df: &DataFrame) -> Vec<Vec<f32>> {
        let mut output_matrix = vec![vec![]; df.len()];
        let data = df.data();
        for (column_name, (dtype, values)) in data {
            match dtype {
                DataType::Float => {
                    let median = df.median(column_name);
                    for (i, value) in values.iter().enumerate() {
                        match value {
                            DataTypeValue::Float(inner) => {
                                output_matrix[i].push(*inner);
                            }
                            DataTypeValue::Null => match self.imputer_strategy {
                                ImputerStrategy::Median => {
                                    output_matrix[i].push(median);
                                }
                            },
                            _ => {}
                        }
                    }
                }
                DataType::String => match self.string_encoding {
                    StringEncoding::OneHot => {
                        let categorical_values = self.extract_categorical_values(values);
                        for (_, cat_values) in categorical_values.iter() {
                            for (i, cat_value) in cat_values.iter().enumerate() {
                                output_matrix[i].push(*cat_value);
                            }
                        }
                    }
                },
                _ => {}
            }
        }
        return output_matrix;
    }

    fn extract_categorical_values(&self, values: &Vec<DataTypeValue>) -> HashMap<String, Vec<f32>> {
        let mut categories: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, value) in values.iter().enumerate() {
            let inner = match value {
                DataTypeValue::String(inner) => inner.clone(),
                DataTypeValue::Null => "null".to_string(),
                _ => {
                    panic!("implementation error")
                }
            };
            for (category, cat_values) in categories.iter_mut() {
                if *category == *inner {
                    cat_values.push(1.0);
                } else {
                    cat_values.push(0.0);
                }
            }
            if !categories.contains_key(inner.as_str()) {
                let mut cat_values = vec![0.0; i];
                cat_values.push(1.0);
                categories.insert(inner.clone(), cat_values);
            }
        }
        return categories;
    }

    // fn scale_data(&self, matrix: &mut Vec<Vec<f32>>) {
    //     match self.scalar {
    //         Scalar::Standard {
    //         },
    //         Scalar::None => {
    //         }
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_df_to_matrix() {
        let imputer_strategy = ImputerStrategy::Median;
        let string_encoding = StringEncoding::OneHot;
        let scalar = Scalar::Standard;
        let pipeline = Pipeline::new(string_encoding, imputer_strategy, scalar);
        let df = DataFrame::from_csv("housing.csv", Some(1000000));
        let output_matrix = pipeline.df_to_matrix(&df);
    }
}
