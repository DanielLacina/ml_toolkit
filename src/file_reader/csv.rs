use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, prelude::*};
use crate::dataframe::{DataFrame, DataType, DataTypeValue};

pub fn df_from_csv(filename: &str, row_limit: Option<usize>) -> DataFrame {
    let mut df = DataFrame::new();
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut reader_lines = reader.lines();
    let headers = reader_lines.next().unwrap().unwrap();
    for header in headers.split(",") {
        df.insert_column(header, &Vec::new(), &DataType::Float); 
    }
    let row_limit = if let Some(limit) = row_limit {
        limit
    } else {
        1000000
    };
    for (i, line) in reader_lines.enumerate() {
        if i == row_limit {
            break;
        }
        let line = line.unwrap();
        insert_row_as_string(&mut df, &line);
    }
    return df;
}


fn insert_row_as_string(df: &mut DataFrame, line: &str) {
    let mut data_hashmap = HashMap::new();
    for (i, value) in line.split(",").enumerate() {
        let (column_name, dtype ) = {
            let (column_name, dtype, _) = df.get_column_by_index(i);
            (column_name.clone(), dtype.clone())
        };
        if value == "" {
            data_hashmap.insert(column_name, DataTypeValue::Null);
        } else {
            match dtype {
                DataType::Float => match value.parse::<f32>() {
                    Ok(parsed_value) => {
                        data_hashmap.insert(column_name, DataTypeValue::Float(parsed_value));
                    }
                    Err(_) => {
                        df.convert_column_values_to_string(column_name.clone().as_str());
                        data_hashmap.insert(column_name, DataTypeValue::String(value.to_string()));
                    }
                },
                DataType::String => {
                    data_hashmap.insert(column_name, DataTypeValue::String(value.to_string()));
                }
                _ => {}
            }
        }
    }
}

