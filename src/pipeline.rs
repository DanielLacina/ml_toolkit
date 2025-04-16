use crate::dataframe::{DataFrame, DataTypeValue};

pub enum ImputerStrategy {
    Median,
}

pub enum StringEncoding {
    OneHot,
}

pub struct Pipeline {
    string_encoding: StringEncoding,
    imputer_strategy: ImputerStrategy,
}

impl Pipeline {
    pub fn new(string_encoding: StringEncoding, imputer_strategy: ImputerStrategy) -> Self {
        Self {
            string_encoding,
            imputer_strategy,
        }
    }
    // pub fn df_to_matrix(&self, df: DataFrame) {
    //     let mut matrix = Vec::new();
    //     let data = df.data().clone();
    //     for row in data {
    //         let mut row_vector = Vec::new();
    //         for value in row.iter() {
    //             match value {
    //                 DataTypeValue::Float(_float) => {
    //                     row_vector.push(_float.clone());
    //                 }
    //             }
    //         }
    //     }
    // }
}
