use crate::{dataframe::{DataFrame, DataType, DataTypeValue}, pipeline::transformers::Transformer};

#[derive(Clone)]
pub enum ImputerStrategy {
    Median,
}

pub struct Imputer {
    strategy: ImputerStrategy,
}

impl Imputer {
    pub fn new(strategy: &ImputerStrategy) -> Self {
        Self {
            strategy: strategy.clone(),
        }
    }

    
}

impl Transformer for Imputer {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df = df.clone();
        let df_column_names: Vec<String> = df
            .columns()
            .into_iter()
            .filter(|df_column_name| {
                *df_column_name != DataFrame::id_column() && column_names.contains(*df_column_name)
            })
            .map(|column_name| column_name.clone())
            .collect();
        for df_column_name in df_column_names.iter() {
            let (dtype, _) = df.get_column(df_column_name);
            match dtype {
                DataType::Float => {
                    let median = df.median(df_column_name);
                    for i in 0..df.len() {
                        let current_value = df.get_cell_value(df_column_name, i).clone();
                        match current_value {
                            DataTypeValue::Float(_) => {}
                            DataTypeValue::Null => {
                                df.modify_cell(df_column_name, i, DataTypeValue::Float(median));
                            }
                            _ => panic!("value type is inconsistent with column datatype header"),
                        }
                    }
                }
                DataType::String => {}
                _ => {
                    panic!("id datatypes cannot be processed")
                }
            }
        }
        return df;
    }
}
