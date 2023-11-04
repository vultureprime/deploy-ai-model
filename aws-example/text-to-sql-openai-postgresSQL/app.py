import os 
import dotenv
from fastapi import FastAPI
from sqlalchemy import text, Table, Column, Integer, String, MetaData, Float
from sqlalchemy.engine import create_engine
from sqlalchemy.sql import insert
from llama_index import SQLDatabase,ServiceContext
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from llama_index.llms import OpenAI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker
from faker import Faker
import random
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


def generate_random_data():
    fake = Faker()

    id = random.randint(1, 1000000)
    name = fake.first_name()
    lastname = fake.last_name()
    height = np.round(random.uniform(140, 220), 2)  # assuming height is in meters
    weight = np.round(random.uniform(50, 100), 2)   # assuming weight is in kilos
   
    return {"id": id, "name": name, "lastname": lastname, "height": height, "weight": weight}


dotenv.load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
HOST = os.environ['HOST']
DBPASSWORD = os.environ['DBPASSWORD']
DBUSER = os.environ['DBUSER']
DBNAME = os.environ['DBNAME']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/helloworld")
async def helloworld():
    return {"message": "Hello World"}

@app.get("/createTable")
async def createTable():
    conn_str = "postgresql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(
        conn_str.format(
            user = DBUSER,
            host = HOST,
            port = '5432',
            password = DBPASSWORD,
            database = DBNAME)
    )
    meta = MetaData()

    students = Table(
        'students', meta, 
        Column('id', Integer, primary_key = True), 
        Column('name', String), 
        Column('lastname', String), 
        Column('height', Float), 
        Column('weight', Float)
    )
    meta.create_all(engine)

    json_compatible_item_data = jsonable_encoder({'message' : 'complete'})
    return JSONResponse(content=json_compatible_item_data)

@app.get("/removeTable")
async def removeTable():

    conn_str = "postgresql://{user}:{password}@{host}:{port}/{database}"

    engine = create_engine(
        conn_str.format(
            user = DBUSER,
            host = HOST,
            port = '5432',
            password = DBPASSWORD,
            database = DBNAME)
    )
    Session = sessionmaker(bind=engine)

    # Create insertion SQL
    with Session() as session :
        session.execute(text("""DROP TABLE "students" """))
        session.commit()

    json_compatible_item_data = jsonable_encoder({'message' : 'complete'})
    return JSONResponse(content=json_compatible_item_data)

@app.get("/getInfo")
async def getInfo():

    conn_str = "postgresql://{user}:{password}@{host}:{port}/{database}"

    engine = create_engine(
        conn_str.format(
            user = DBUSER,
            host = HOST,
            port = '5432',
            password = DBPASSWORD,
            database = DBNAME)
    )

    metadata = MetaData()
    metadata.reflect(bind=engine)
    result = {
        'attr' : {}
    }
    for table in metadata.tables:
        print("Table name: ", table)
        print("Table details: ")
        # retrieving table details
        result['table_name'] = table
        for column in metadata.tables[table].c:
            result['attr'][column.name] = {
                'name' : column.name,
                'type' : str(column.type)
            }
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/addRandomData")
async def addRandomData():

    conn_str = "postgresql://{user}:{password}@{host}:{port}/{database}"
    data = []
    for i in range(10):
        data.append(generate_random_data())

    engine = create_engine(
        conn_str.format(
            user = DBUSER,
            host = HOST,
            port = '5432',
            password = DBPASSWORD,
            database = DBNAME)
    )
    metadata = MetaData()
    metadata.reflect(bind=engine)
    students = Table('students', metadata)

    Session = sessionmaker(bind=engine)

    # Create insertion SQL
    stmt = (insert(students).values(data))
    with Session() as session :
        session.execute(stmt)
        session.commit()

    print("All records inserted.")
    json_compatible_item_data = jsonable_encoder(data)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/getAllData")
async def getAllData():

    conn_str = "postgresql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(
        conn_str.format(
            user = DBUSER,
            host = HOST,
            port = '5432',
            password = DBPASSWORD,
            database=DBNAME)
    )
    sql_database = SQLDatabase(engine)
    res = sql_database.run_sql("SELECT * FROM students")

    result = {}
    for i in res[1]['result'] : 
        result[i[0]] = {
            'id' : i[0],
            'name' : i[1],
            'lastname' : i[2],
            'height' : i[3],
            'weight' : i[4]
        }
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)



@app.get("/queryWithPrompt")
async def queryWithPrompt(query_str : str = 'Write SQL in PostgresSQL format. Get average hight of student.'):

    llm = OpenAI(model="gpt-4",temperature=0, max_tokens=256)

    conn_str = "postgresql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(
        conn_str.format(
            user = DBUSER,
            host = HOST,
            port = '5432',
            password = DBPASSWORD,
            database=DBNAME)
    )
    service_context = ServiceContext.from_defaults(
        llm=llm
    )
    sql_database = SQLDatabase(engine)

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        service_context=service_context
    )

    query_prompt = (query_str)
    try : 
        response = query_engine.query(query_prompt)

        result = {
            "result": response.response,
            "SQL Query": response.metadata['sql_query']
        }
        
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)


    except :
        result = {
            "result": 'Bad Prompt',
            "SQL Query": 'Null'
        }
        
        json_compatible_item_data = jsonable_encoder(result)
        return JSONResponse(content=json_compatible_item_data)