

def create_table(cursor, table_name: str, schema: str):
    """ create table_name if dose not exist in database 

    Args:
        cursor : database.cursor
        table_name (str): name of table
        schema (str): schema of expected table
    """
    cursor.execute(f"SHOW TABLES")
    fetchs = cursor.fetchall()
    if len(fetchs) !=0:
        tables = fetchs[0]
        if table_name not in tables:
            print(f"  mysql>> create table: {table_name}")
            cursor.execute(schema)
        else:
            print(f"  mysql>> table: {table_name} is already exist!")
    else:
        print(f"  mysql>> create table: {table_name}")
        cursor.execute(schema)