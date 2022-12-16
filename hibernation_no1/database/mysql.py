

def create_table(cursor, table_name: str, schema: str):
    """ create table_name if dose not exist in database 

    Args:
        cursor : database.cursor
        table_name (str): name of table
        schema (str): schema of expected table
    """
    cursor.execute(f"SHOW TABLES")
    fetchs = cursor.fetchall()
    
    tables = []
    if len(fetchs) !=0:
        for fetch in fetchs:
            tables.append(fetch[0])
    else:
        print(f"  mysql>> create table: {table_name}")
        cursor.execute(schema)
    
    if len(fetchs) !=0:
        if table_name not in tables:
            print(f"  mysql>> create table: {table_name}")
            cursor.execute(schema)
        else:
            print(f"  mysql>> table: {table_name} is already exist!")      
            


def check_table_exist(cursor, tables_cfg):
    """ check tables are exist in database

    Args:
        cursor : pymysql.connect.cursor  
        tables_cfg (dict or list or str): table names
    """
    if isinstance(tables_cfg, dict):
        table_names = []
        for _, name in tables_cfg.items():
            table_names.append(name)
    elif isinstance(tables_cfg, list):
        table_names = tables_cfg
    elif isinstance(table_names, str):
        table_names = [table_names]
        
    else: raise TypeError(f" `tables_cfg` type must be dict or list or str!")
    
    cursor.execute(f"SHOW TABLES")
    fetchs = cursor.fetchall()
    
    tables = []
    if len(fetchs) !=0:
        for fetch in fetchs:
            tables.append(fetch[0])
    else:
        raise AttributeError(f"Table does not exist in the database!")

    for table_name in table_names :
        if table_name not in tables:
            raise AttributeError(f"Table: {table_name} is not exist in database!")