# Call Center NBA Demos

## What is included

### Notebooks  

**Make sure to edit the `catalog` and `schema` variable names in the notebooks**
  
**Call Center Call Transcribe & Score**
- `call_center_transcribe_demo.py`: this shows a basic demonstration of using OpenAI to first transcribe the MP3 file in a UC Volume and then use AI_QUERY() to do the scoring based on a prompt constructed from the scorecard  
  - You need to upload the audio file in the `audio_files` folder to a UC Volume
  
**Identify Engagement Propensity & Next Best Action**
- `call_center_demo_setup.py`: set up the initial datasets for the demo
- `call_center_ml_model.py`: leverage XGBoost and LightGBM models on the structured data to build models that help identify cohorts of engagement and what the next best action for the type of cohort
- `ml_concepts_explained.md`: explanation of why Classic ML models perform better for this

