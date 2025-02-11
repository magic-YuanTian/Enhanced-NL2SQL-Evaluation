import sqlite3
import uuid
import os
from collections import Counter, defaultdict
import random
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Union
from sql_metadata import Parser
import json
import copy


# Helper functions
def generate_uuid():
    return str(uuid.uuid4())

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def random_int():
    return random.randint(1, 9999)

def random_string(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def random_boolean():
    return random.choice([True, False])

def random_timestamp():
    return random_date(datetime(2020, 1, 1), datetime.now()).isoformat()

def generate_new_value(field_type):
    if field_type == 'int' or field_type == 'integer':
        return random_int()
    elif field_type == 'text':
        return random_string('value')
    elif field_type == 'boolean':
        return random_boolean()
    elif field_type == 'timestamp' or field_type == 'date':
        return random_timestamp()
    else:
        return random_string('value')

def generate_value(field, foreign_key_values, existing_values, repetition_prob=0.3):
    field_type = field.get('type', 'text')
    if field.get('foreign_ref'):
        referenced_table, referenced_field = field['foreign_ref'].split('(')
        referenced_field = referenced_field.rstrip(')')
        foreign_key = (referenced_table, referenced_field)
        if foreign_key in foreign_key_values and foreign_key_values[foreign_key]:
            return random.choice(foreign_key_values[foreign_key])
        else:
            # If no foreign key values are available, generate a new value
            return generate_new_value(field_type)
    
    # Check if we should use an existing value
    if not field.get('isPrimary') and existing_values and random.random() < repetition_prob:
        return random.choice(existing_values)
    
    return generate_new_value(field_type)

def determine_table_order(schema):
    dependencies = defaultdict(set)
    for table, fields in schema.items():
        for field in fields:
            if field.get('foreign_ref'):
                referenced_table = field['foreign_ref'].split('(')[0]
                dependencies[table].add(referenced_table)
    
    ordered_tables = []
    visited = set()
    
    def dfs(table):
        if table in visited:
            return
        visited.add(table)
        for dep in dependencies[table]:
            dfs(dep)
        ordered_tables.append(table)
    
    for table in schema:
        dfs(table)
    
    return ordered_tables



def extract_values_and_columns_from_select(sql_query: str, schema: dict) -> Dict[str, List[Tuple[str, Union[str, Dict[str, str]]]]]:
    # Extract table names and their aliases
    table_pattern = r'\bfrom\s+([\w\s,`]+)(?:\bwhere|\bjoin|\bgroup by|\border by|\blimit|\Z)'
    table_match = re.search(table_pattern, sql_query, re.IGNORECASE)
    tables = {}
    if table_match:
        table_clause = table_match.group(1)
        # Updated pattern to handle backticks and dots in table names
        table_parts = re.findall(r'([`\w.]+)(?:\s+(?:as\s+)?(\w+))?', table_clause)
        tables = {alias or name.replace('`', ''): name.replace('`', '') for name, alias in table_parts}
    
    # Define regex patterns for different parts of the SELECT statement
    where_pattern = r'\bwhere\s+(.*?)(?:\bgroup by|\border by|\blimit|\Z)'
    
    result: Dict[str, List[Tuple[str, Union[str, Dict[str, str]]]]] = defaultdict(list)
    
    # Extract from WHERE clause
    where_match = re.search(where_pattern, sql_query, re.IGNORECASE)
    if where_match:
        where_conditions = re.split(r'\s+and\s+', where_match.group(1), flags=re.IGNORECASE)
        for condition in where_conditions:
            # Updated pattern to handle backticks and more operators
            match = re.match(r'([`\w.]+)\s*(=|!=|<>|>|<|>=|<=|like|in)\s*([^,;\s]+)', condition, re.IGNORECASE)
            if match:
                column, operator, value = match.groups()
                
                # Skip nested queries as values
                if value.startswith('(') or 'select ' in value.lower():
                    continue
                
                # Clean column name (remove backticks)
                column = column.replace('`', '')
                
                # Try to find the actual table and column
                try:
                    table, col = get_table_and_column(column, tables, schema)
                    if table and table in schema:  # Verify table exists in schema
                        actual_table = tables.get(table, table)
                        # Verify column exists in schema
                        if any(c['field'].lower() == col.lower() for c in schema[actual_table]['columns']):
                            result[actual_table].append((col, clean_value(value)))
                except Exception as e:
                    # Log the error for debugging if needed
                    # print(f"Error processing column {column}: {str(e)}")
                    continue
    
    return dict(result)


# Given a schema and a column, find the table that contains the column
def find_table_from_column(schema, column):
    for table in schema:
        for col in schema[table]['columns']:
            if col['field'] == column:
                return table



def clean_value(value: str) -> str:
    """Clean and normalize SQL values."""
    value = value.strip()
    # Remove quotes if present
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    # Handle boolean values
    if value.lower() in ('true', '1'):
        return '1'
    elif value.lower() in ('false', '0'):
        return '0'
    return value


def get_table_and_column(full_column: str, tables: Dict[str, str], schema: dict) -> Tuple[str, str]:
    parts = full_column.split('.')
    if len(parts) == 2:
        table_alias, column = parts
        # Try to find the actual table name from aliases
        table_name = tables.get(table_alias, table_alias)
        # Verify the table exists in schema
        if table_name in schema:
            # Case-insensitive column matching
            for col in schema[table_name]['columns']:
                if col['field'].lower() == column.lower():
                    return table_name, col['field']  # Return the actual column name from schema
    
    # If no table prefix or table not found, search all tables
    column_name = parts[-1]
    for table_name, table_info in schema.items():
        # Case-insensitive column matching
        for col in table_info['columns']:
            if col['field'].lower() == column_name.lower():
                return table_name, col['field']  # Return the actual column name from schema
    
    # If still not found, return the original table alias (if exists) or None
    return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])
        

def determine_table_order(schema):
    dependencies = defaultdict(set)
    for table, fields in schema.items():
        if isinstance(fields, dict) and 'columns' in fields:
            fields = fields['columns']
        for field in fields:
            if isinstance(field, dict) and field.get('foreign_ref'):
                referenced_table = field['foreign_ref'].split('(')[0]
                dependencies[table].add(referenced_table)
            elif isinstance(field, str):
                # If field is a string, we can't determine dependencies
                # so we'll just skip it
                continue
    
    ordered_tables = []
    visited = set()
    
    def dfs(table):
        if table in visited:
            return
        visited.add(table)
        for dep in dependencies[table]:
            dfs(dep)
        ordered_tables.append(table)
    
    for table in schema:
        dfs(table)
    
    return ordered_tables



def generate_data(schema, num_records=10, extracted_values=None):
    try:
        data = {}
        foreign_key_values = defaultdict(list)
        existing_values = defaultdict(list)
        
        try:
            table_order = determine_table_order(schema)
        except Exception as e:
            raise Exception(f"Failed to determine table order. Schema structure may be invalid: {str(e)}")

        # Preprocess schema
        try:
            for key in schema.keys():    
                if isinstance(schema[key], dict) and 'columns' in schema[key]:
                    schema[key] = schema[key]['columns']
        except Exception as e:
            raise Exception(f"Invalid schema structure for table '{key}': {str(e)}")

        # Create mappings
        field_case_mapping = {}
        field_order = {}
        field_type_mapping = {}
        
        try:
            for table, fields in schema.items():
                field_case_mapping[table] = {}
                field_order[table] = []
                field_type_mapping[table] = {}
                
                for field in fields:
                    try:
                        if isinstance(field, dict):
                            field_name = field['field']
                            field_type = field.get('type', 'text').lower()
                        else:
                            field_name = field
                            field_type = 'text'
                        
                        field_case_mapping[table][field_name.lower()] = field_name
                        field_order[table].append(field_name)
                        field_type_mapping[table][field_name] = field_type
                    except Exception as e:
                        raise Exception(f"Invalid field definition in table '{table}': {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing schema fields: {str(e)}")

        def create_ordered_record(table_name, field_values=None):
            if field_values is None:
                field_values = {}
            record = {}
            
            try:
                for field in field_order[table_name]:
                    try:
                        field_info = {
                            'field': field,
                            'type': field_type_mapping[table_name][field]
                        }
                        
                        original_field = next(
                            (f for f in schema[table_name] 
                             if isinstance(f, dict) and f['field'] == field),
                            None
                        )
                        if original_field and 'foreign_ref' in original_field:
                            field_info['foreign_ref'] = original_field['foreign_ref']
                        
                        lower_field = field.lower()
                        
                        if lower_field in field_values:
                            value = field_values[lower_field]
                        else:
                            try:
                                value = generate_value(
                                    field_info,
                                    foreign_key_values,
                                    existing_values.get(lower_field, [])
                                )
                            except Exception as e:
                                raise Exception(f"Failed to generate value for field '{field}' of type '{field_info['type']}': {str(e)}")
                        
                        record[field] = value
                    except Exception as e:
                        raise Exception(f"Error processing field '{field}' in table '{table_name}': {str(e)}")
            except Exception as e:
                raise Exception(f"Failed to create record for table '{table_name}': {str(e)}")
            
            return record

        # Generate initial data
        for table_name in table_order:
            try:
                table_data = []
                for i in range(num_records):
                    try:
                        record = create_ordered_record(table_name)
                        if 'id' not in record:
                            record['id'] = generate_uuid()
                        table_data.append(record)
                        
                        for column, value in record.items():
                            foreign_key_values[(table_name, column.lower())].append(value)
                            existing_values[column.lower()].append(value)
                    except Exception as e:
                        raise Exception(f"Failed to generate record {i+1}/{num_records} for table '{table_name}': {str(e)}")
                
                data[table_name] = table_data
            except Exception as e:
                raise Exception(f"Failed to generate data for table '{table_name}': {str(e)}")

        # Handle extracted values
        if extracted_values:
            try:
                added_value = []
                for used_table in extracted_values.keys():
                    if used_table not in data:
                        print('------------- data\n', data)
                        print('------------- extracted_values\n', extracted_values)
                        raise Exception(f"Table '{used_table}' from extracted values not found in schema")
                        
                    for column, value in extracted_values[used_table]:
                        if column.lower() not in field_case_mapping[used_table]:
                            print(f"Table: {used_table}")
                            print(f"Column: {column}")
                            print(f"Available columns: {list(field_case_mapping[used_table].keys())}")
                            print(f"Field case mapping: {field_case_mapping[used_table]}")
                            raise Exception(f"Column '{column}' not found in table '{used_table}'")
                            
                        attempt = 0
                        success = False
                        while attempt < 10:
                            attempt += 1
                            if len(data[used_table]) == 0:
                                raise Exception(f"No records available in table '{used_table}'")
                            
                            record_index = random.randint(0, len(data[used_table]) - 1)
                            
                            if data[used_table][record_index][column] not in added_value:
                                data[used_table][record_index][column] = value
                                added_value.append(value)
                                success = True
                                break
                                
                        if not success:
                            raise Exception(f"Failed to insert extracted value '{value}' for column '{column}' in table '{used_table}' after {attempt} attempts")
                            
            except Exception as e:
                raise Exception(f"Error processing extracted values: {str(e)}")

        return data

    except Exception as e:
        raise Exception(f"Data generation failed: {str(e)}")

def add_extracted_values(data, extracted_values):
    for table_name, columns in extracted_values.items():
        if table_name not in data:
            print(f"Warning: Table '{table_name}' not found in generated data. Creating new table.")
            data[table_name] = []
        
        # Create a new record with extracted values
        new_record = {}
        for column, value in columns:
            new_record[column] = value
        
        # Add any missing fields with random values
        for field in data[table_name][0].keys() if data[table_name] else []:
            if field not in new_record:
                new_record[field] = generate_value({'type': 'text'}, {}, [])
        
        # Ensure the record has an ID
        if 'id' not in new_record:
            new_record['id'] = generate_uuid()
        
        # Add the new record to the table
        data[table_name].append(new_record)
        # print(f"Added record with extracted values to {table_name}: {new_record}")
    
    return data

def create_database(data, db_dir):
    # check if db_dir exists
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    random_id = uuid.uuid4().hex[:6]
    db_path = os.path.join(db_dir, f"records_{random_id}.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path, timeout=20) as conn:
        cursor = conn.cursor()

        for table_name, records in data.items():
            if records:
                # Use dict.fromkeys to preserve order while removing duplicates
                columns = list(dict.fromkeys(records[0].keys()))
                
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'[{col}] TEXT' for col in columns])})"
                cursor.execute(create_table_sql)

                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO {table_name} ({', '.join([f'[{col}]' for col in columns])}) VALUES ({placeholders})"

                for record in records:
                    values = [str(record.get(col, '')) for col in columns]
                    cursor.execute(insert_sql, values)

        conn.commit()

    return db_path

def execute_query(query, db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # print(f"Executing query: {query}")
        
        # print('*'*40)
        # print(query)
        # print('*'*40)
        
        cursor.execute(query)
        
        
        column_names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        # print(f"Query result: {rows}")
        result = [dict(zip(column_names, row)) for row in rows]
        
        table_name = "result"
        from_parts = query.upper().split(' FROM ')
        if len(from_parts) > 1:
            table_parts = from_parts[1].strip().split()
            if table_parts:
                table_name = table_parts[0].lower()
        
        return {table_name: result}





def compare_results(gold_result, predicted_result):
    gold_table = list(gold_result.keys())[0]
    pred_table = list(predicted_result.keys())[0]

    gold_rows = gold_result[gold_table]
    pred_rows = predicted_result[pred_table]

    # Handle empty result sets
    if not gold_rows and not pred_rows:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "result_exact_match": True,
            "column_match": True,
            "data_match": True,
            "column_coverage": 1.0,
            "gold_count": 0,
            "predicted_count": 0,
            "correct_count": 0,
            "gold_columns": set(),
            "pred_columns": set(),
            "common_columns": set(),
            "all_columns": set()
        }

    # Handle cases where one result set is empty
    if not gold_rows or not pred_rows:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "result_exact_match": False,
            "column_match": False,
            "data_match": False,
            "column_coverage": 0.0,
            "gold_count": len(gold_rows),
            "predicted_count": len(pred_rows),
            "correct_count": 0,
            "gold_columns": set(gold_rows[0].keys()) if gold_rows else set(),
            "pred_columns": set(pred_rows[0].keys()) if pred_rows else set(),
            "common_columns": set(),
            "all_columns": set(gold_rows[0].keys() if gold_rows else []).union(set(pred_rows[0].keys() if pred_rows else []))
        }

    gold_columns = set(gold_rows[0].keys()) if gold_rows else set()
    pred_columns = set(pred_rows[0].keys()) if pred_rows else set()
    common_columns = gold_columns.intersection(pred_columns)
    all_columns = gold_columns.union(pred_columns)

    # Calculate column coverage
    column_coverage = len(common_columns) / len(all_columns) if all_columns else 0

    # Compare rows based on common columns
    gold_set = set(frozenset((k, v) for k, v in row.items() if k in common_columns) for row in gold_rows)
    pred_set = set(frozenset((k, v) for k, v in row.items() if k in common_columns) for row in pred_rows)

    intersection = gold_set.intersection(pred_set)

    # Calculate base precision and recall
    base_precision = len(intersection) / len(pred_set) if pred_set else 0
    base_recall = len(intersection) / len(gold_set) if gold_set else 0

    # Apply column coverage to get final precision and recall
    precision = base_precision * column_coverage
    recall = base_recall * column_coverage

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    column_match = gold_columns == pred_columns
    data_match = gold_set == pred_set
    exact_match = column_match and data_match

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "result_exact_match": exact_match,
        "column_match": column_match,
        "data_match": data_match,
        "column_coverage": column_coverage,
        "gold_count": len(gold_rows),
        "predicted_count": len(pred_rows),
        "correct_count": len(intersection),
        "gold_columns": gold_columns,
        "pred_columns": pred_columns,
        "common_columns": common_columns,
        "all_columns": all_columns
    }


# the interface function that can be called from outside, directly comparing the gold and predicted sql and returned the enhanced execution accuracy
def enhanced_execution_accuracy(sql_gold, sql_predicted, schema, num_records=1000):
    
    sql_gold_ori = copy.deepcopy(sql_gold)
    sql_predicted_ori = copy.deepcopy(sql_predicted)
    

    try:
        sql_gold = removeAlias(sql_gold)
        sql_predicted = removeAlias(sql_predicted)
    except Exception as e:
        raise Exception(f"\n* Error in removing alias: {str(e)}\n... *")
    
    


    # print('--- after removing alias ---')
    print("Gold SQL:", sql_gold)
    print("Predicted SQL:", sql_predicted)
    


    # Extract values from both queries
    try:
        extracted_values_gold = extract_values_and_columns_from_select(sql_gold, schema)
    except Exception as e:
        raise Exception(f"\n* Error in extracting gold values: {str(e)}\n... *")
    


    try:
        extracted_values_predicted = extract_values_and_columns_from_select(sql_predicted, schema)
    except Exception as e:
        raise Exception(f"\n* Error in extracting predicted values: {str(e)}\n... *")
    # print("Extracted values (predicted):", extracted_values_predicted)



    # Combine extracted values
    combined_extracted_values = {}
    for table in set(extracted_values_gold.keys()) | set(extracted_values_predicted.keys()):
        combined_extracted_values[table] = (
            extracted_values_gold.get(table, []) +
            extracted_values_predicted.get(table, [])
        )
    
    # Remove duplicate value pairs from combined extracted values
    for table in combined_extracted_values:
        # Convert list of tuples to set to remove duplicates, then back to list
        combined_extracted_values[table] = list(set(combined_extracted_values[table]))

    # print("Combined extracted values:")
    # print(combined_extracted_values)
    

    # print('x'*40)
    # print(combined_extracted_values)
    # print('x'*40)

    # # Generate synthetic data with extracted values
    # print("Generating synthetic data...")
    try:
        synthetic_records = generate_data(schema, num_records=num_records, extracted_values=combined_extracted_values)
        cleaned_records = clean_synthetic_records(synthetic_records)
        db_path = create_database(cleaned_records, 'temp_db')
    except Exception as e:
        raise Exception(f"\n* Error in generating synthetic data: {str(e)}\n... *")
    


    
    # print("Creating database...", flush=True)
    try:
        # print("Executing SQL queries...", flush=True)

        # check whether the gold sql is valid, if it is not, there is no need to continue, return 0

        try:
            result_gold = execute_query(sql_gold_ori, db_path)

        except Exception as e:

            print(f"\n* Error in gold SQL query: {str(e)}\nSkipping evaluation... *")

            return 0

        try:
            result_predicted = execute_query(sql_predicted_ori, db_path)
        except Exception as e:
            print(f"\n* Error in predicted SQL query: {str(e)}\nSkipping evaluation... *")
            return -1
        

        # output 2 results
        # print("\nGold Result --------------->")
        # print(result_gold)
        # print("\nPredicted Result --------------->")
        # print(result_predicted)
        
        # print("Comparing results...", flush=True)
        comparison = compare_results(result_gold, result_predicted)

        # print("\nExecution Accuracy Results:")
        # print(f"Precision: {comparison['precision']:.2f}")
        # print(f"Recall: {comparison['recall']:.2f}")
        # print(f"F1 Score: {comparison['f1_score']:.2f}")
        # print(f"Exact Match: {'Yes' if comparison['result_exact_match'] else 'No'}")
        # print(f"Column Match: {'Yes' if comparison['column_match'] else 'No'}")
        # print(f"Data Match: {'Yes' if comparison['data_match'] else 'No'}")
        # print(f"Gold Result Count: {comparison['gold_count']}")
        # print(f"Predicted Result Count: {comparison['predicted_count']}")
        # print(f"Correct Predictions: {comparison['correct_count']}")
        # print(f"Gold Columns: {comparison['gold_columns']}")
        # print(f"Predicted Columns: {comparison['pred_columns']}")
        # print(f"Common Columns: {comparison['common_columns']}")
        
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                # print(f"\nDeleted temporary database: {db_path}")
            except Exception as e:
                print(f"\nError deleting database file: {e}")
        else:
            print(f"\nDatabase file not found: {db_path}")
    
    # empty the temp_db folder
    for file in os.listdir('temp_db'):
        os.remove(os.path.join('temp_db', file))
    
    return comparison




def removeAlias(sql):
    # Store the original SQL for case preservation
    original_sql = sql
    
    # Work with lowercase for case-insensitive matching
    sql_lower = sql.lower()
    
    # Remove space around '.'
    sql_lower = re.sub(r' *\. *', '.', sql_lower)
    
    # Detect all the table AS xxx as alias using regular expression
    alias_dict = {}
    for match in re.finditer(r'(\w+) +as +(\w+)\b', sql_lower):
        alias_dict[match.group(2)] = match.group(1)
    
    # Function to replace aliases in a case-insensitive manner
    def replace_alias(match):
        full_match = match.group(0)
        alias = match.group(1)
        if alias.lower() in alias_dict:
            return alias_dict[alias.lower()] + '.'
        return full_match

    # Replace aliases in column references
    result = re.sub(r'(\w+)\.', replace_alias, original_sql)
    
    # Remove AS clauses
    result = re.sub(r'\b(\w+) +AS +\w+\b', r'\1', result, flags=re.IGNORECASE)
    
    # Remove standalone aliases
    for alias in alias_dict:
        result = re.sub(r'\b' + re.escape(alias) + r'\b', '', result, flags=re.IGNORECASE)
    
    # Remove extra spaces
    result = re.sub(r' +', ' ', result)
    
    return result.strip()


# convert schema format
def convert_schema_format(input_schema):
    
    def get_foreign_key_referred_table_column(string):
        
        if 'foreign key' not in string.lower():
            return string
        
        # Regex pattern to extract table name
        tab_pattern = r"references\s+(\w+)\s*\("        
        # Regex pattern to extract column name
        col_pattern = r"foreign\s+key\s*\(([^)]+)\)"
        # Regex pattern to extract external column name
        ext_col_pattern = r"references\s+\w+\s*\(([^)]+)\)"

        # Extract names
        referenced_table = re.search(tab_pattern, string, re.IGNORECASE)
        this_column = re.search(col_pattern, string, re.IGNORECASE)
        referenced_column = re.search(ext_col_pattern, string, re.IGNORECASE)
        
        if referenced_table and this_column and referenced_column:
            return (
                this_column.group(1).strip(),
                referenced_table.group(1),
                referenced_column.group(1).strip()
            )
        else:
            raise ValueError("Unable to parse foreign key string")
    
    # get table list
    tables = list(input_schema.keys())
    
    for table in input_schema:
        for i in range(len(input_schema[table]['columns'])):
            if 'foreign key' in input_schema[table]['columns'][i]['field']:
                this_column, referenced_table, referenced_column = get_foreign_key_referred_table_column(input_schema[table]['columns'][i]['field'])
                input_schema[table]['columns'][i]['field'] = this_column
                # if the referenced table is not in the schema, ignore it
                if referenced_table not in tables:
                    input_schema[table]['columns'][i]['foreign_ref'] = None
                else:
                    input_schema[table]['columns'][i]['foreign_ref'] = f"{referenced_table}({referenced_column})"
            else:
                input_schema[table]['columns'][i]['foreign_ref'] = None
                
    return input_schema


def eval_enhanced_exec_acc(data_path):


    # read json schema file and convert to dictionary
    with open(data_path) as f:
        schema = json.load(f)


    results = []
    predict_error_cnt = 0
    comparison_success_cnt = 0
    correct_cnt = 0

    other_error_cnt = 0

    cnt = 0
    start_idx = 1

    gold_error_cnt = 0

    for instance in schema:
        cnt += 1
        
        if cnt < start_idx:
            continue
        
        print('--------')

        print(f"prediction Execution Errors: {predict_error_cnt}")
        print(f"Gold Execution Errors: {gold_error_cnt}")

        print(f"Successfully processed {comparison_success_cnt} instances")
        temp_accuracy = correct_cnt/comparison_success_cnt if comparison_success_cnt > 0 else 0
        print(f"Current Accuracy: {temp_accuracy}")


        print('-'*30 + str(cnt) + '-'*30)
        
        try:
            sql_gold = instance['gold_sql']
            sql_predicted = instance['predicted_sql']
            
            print('Gold SQL:', sql_gold)
            print('Predicted SQL:', sql_predicted)
            
            schema = convert_schema_format(instance['relevant_table_schema'])
            
            # calculate execution accuracy
            result = enhanced_execution_accuracy(sql_gold, sql_predicted, schema, num_records=1000)
            
            # if the gold sql is invalid (result == 0), no need to continue
            if result == 0:
                gold_error_cnt += 1
                continue
            elif result == -1:
                predict_error_cnt += 1
                comparison_success_cnt += 1
                continue
            

            instance['execution_eval'] = result
            
            print('Execution match? ', result['result_exact_match'])
            
            results.append(copy.deepcopy(instance))


        except Exception as e:
            print('\n* Other error happens:', e, ' *\n')
            other_error_cnt += 1
            continue
            

        comparison_success_cnt += 1
        
        # get the number of correct predictions
        if result['result_exact_match']:
            correct_cnt += 1

    
    print('-'*30 + 'Final Statistics' + '-'*30)
    # print the results
    print('Successfully compared instances:', comparison_success_cnt)
    print('Error instances:', predict_error_cnt)

    print('Gold sql error instances:', gold_error_cnt)
    print('total instances:', cnt)


    print('Correct predictions (exact execution match):', str(correct_cnt) + '/' + str(comparison_success_cnt))
    print('Accuracy:', correct_cnt/comparison_success_cnt if comparison_success_cnt > 0 else 0)

    print('-'*60)
    
    final_accuracy = correct_cnt/comparison_success_cnt if comparison_success_cnt > 0 else 0

    return final_accuracy

def clean_synthetic_records(records):
    """Clean synthetic records to ensure no duplicate columns (case-insensitive)."""
    cleaned_data = {}
    
    for table_name, table_records in records.items():
        if not table_records:
            cleaned_data[table_name] = []
            continue
            
        # Get all columns from first record
        columns = list(table_records[0].keys())
        
        # Create case-insensitive column mapping, keeping first occurrence
        seen_columns = {}
        unique_columns = []
        for col in columns:
            col_lower = col.lower()
            if col_lower not in seen_columns:
                seen_columns[col_lower] = col
                unique_columns.append(col)
        
        # Create new records with only unique columns
        cleaned_records = []
        for record in table_records:
            cleaned_record = {col: record[col] for col in unique_columns}
            cleaned_records.append(cleaned_record)
            
        cleaned_data[table_name] = cleaned_records
    
    return cleaned_data