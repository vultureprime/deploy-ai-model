import os 
import boto3
import dotenv
import json
from fastapi import FastAPI
from sqlalchemy import func, select, text
from sqlalchemy.engine import create_engine
from llama_index import SQLDatabase,ServiceContext
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from llama_index.llms import OpenAI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
AWS_REGION = os.environ['AWS_REGION']
S3_STAGING_DIR = os.environ['S3_STAGING_DIR']
DATABASE = os.environ['DATABASE']
WORKGROUP = os.environ['WORKGROUP']
TABLE = os.environ['TABLE']

class Item(BaseModel) : 
    name : str
    type : str

app = FastAPI()
@app.get("/helloworld")
async def helloworld():
    return {"message": "Hello World"}

@app.get("/query")
async def query(query_str : str = 'Write SQL in PrestoSQL format. Get only top 10 data. The data must correct timestamp fomat value from review_creation_date'):
    boto3.client(
        'athena',
        region_name=AWS_REGION,
    )

    llm = OpenAI(model="gpt-4",temperature=0, max_tokens=256)

    conn_str = "awsathena+rest://:@athena.{region_name}.amazonaws.com:443/"\
            "{database}?s3_staging_dir={s3_staging_dir}?work_group={workgroup}"

    engine = create_engine(conn_str.format(
        region_name=AWS_REGION,
        s3_staging_dir=S3_STAGING_DIR,
        database=DATABASE,
        workgroup=WORKGROUP
    ))
    service_context = ServiceContext.from_defaults(
        llm=llm
    )
    sql_database = SQLDatabase(engine, include_tables=[TABLE])

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[TABLE],
        service_context=service_context
    )
    query_prompt = (query_str)
    response = query_engine.query(query_prompt)
    result = {
        "result": response.response,
        "SQL Query": response.metadata['sql_query']
    }
    
    return json.dumps(result)