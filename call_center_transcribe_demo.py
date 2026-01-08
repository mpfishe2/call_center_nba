# Databricks notebook source
# MAGIC %md
# MAGIC # Call Center Demo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transcribe Audio --> Text

# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

from openai import OpenAI

secret_scope = ""

openai_key = dbutils.secrets.get(secret_scope, "openaikey")

openai_client = OpenAI(api_key=openai_key)

catalog = ""
schema = ""

# COMMAND ----------

# improvement: read mp3 as binaryFile to be passed through ai_query?
audio_path = f"/Volumes/{catalog}/{schema}/recordings/1735404531458927.mp3"

with open(audio_path, "rb") as f:
  transcription = openai_client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=f,
    response_format="text",
    temperature=0,
    language="en"
  )
print(transcription)

# COMMAND ----------

transcription_df = spark.createDataFrame([[audio_path, transcription]], ["path", "transcription"])
display(transcription_df)

# COMMAND ----------

transcription_df.write.saveAsTable(f"{catalog}.{schema}.transcriptions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scorecards --> Prompts for LLM Usage

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

COC_Scorecard_Prompt = """
ONLY RETURN ANSWERS IN JSON STRUCT DEFINED BELOW

You are an analyst scoring how well call center support specialists are performing.  You will be given a transcript of a call and the following criteria to judge it by. 

Each criteria has skills within them that you associate a score with.

Criteria 1: Technical Aspects
1. The agent satisfied proper recording disclosure requirements throughout the call.
  - If yes, then they get six points
  - If no, then they get zero points
2. The agent satisfied proper member authentication requirements 
  - If yes, then they get eight points
  - If no, then they get zero points
3. The agent satisfied call closing requirements appropriately 
  - If yes, then they get four points
  - If no, then they get zero points

Criteria 2: Quality of Service
1. The agent maintained professionalism and employed soft skills throughout the call
  - Rate on a scale of 1 to 10 where 1 is unprofessional and 10 is most professional
2. The agent offered correct and accurate program information when necessary
  - Rate on a scale of 1 to 10 with 1 being incorrect and 10 being correct
3. The agent demonstrated a positive and helpful demeanor throughout the call
  - Rate on a scale of 1 to 10 with 1 being negative and 10 being positive

For the response, structure it so that it is a JSON object with the following keys:

- "criteria_1": a dictionary with the following keys:
  - "technical_aspects": a dictionary with the following keys:
    - "recording_disclosure": a dictionary with the following keys:
      - "score": an integer between 0 and 6
    - "member_authentication": a dictionary with the following keys:
      - "score": an integer between 0 and 8
    - "call_closing": a dictionary with the following keys:
      - "score": an integer between 0 and 4
- "criteria_2": a dictionary with the following keys:
  - "quality_of_service": a dictionary with the following keys:
    - "professionalism": a dictionary with the following keys:
      - "score": an integer between 1 and 10
    - "program_information": a dictionary with the following keys:
      - "score": an integer between 1 and 10
    - "demeanor": a dictionary with the following keys:
      - "score": an integer between 1 and 10
- "total_score": an integer between 0 and 48

Your response should ONLY be a JSON object with the above structure.
"""

returnType = """
STRUCT<
  criteria_1: STRUCT<
    technical_aspects: STRUCT<
      recording_disclosure: STRUCT<score: INT>,
      member_authentication: STRUCT<score: INT>,
      call_closing: STRUCT<score: INT>
    >
  >,
  criteria_2: STRUCT<
    quality_of_service: STRUCT<
      professionalism: STRUCT<score: INT>,
      program_information: STRUCT<score: INT>,
      demeanor: STRUCT<score: INT>
    >
  >,
  total_score: INT
>
"""

# COMMAND ----------

## improvement: structured outputs
transcription_scored_df = spark.sql(f"""
SELECT 
  path,
  transcription,
  ai_query(
    'databricks-gpt-oss-20b',
    '{COC_Scorecard_Prompt}\nTranscript: ' || transcription
  ) AS scorecard
FROM maxf_demos.call_center.transcriptions
""")
display(transcription_scored_df.withColumn("scorecard", from_json("scorecard", returnType)))