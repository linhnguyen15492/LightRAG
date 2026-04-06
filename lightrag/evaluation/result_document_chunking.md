INFO: ======================================================================
INFO: 🔍 RAGAS Evaluation - Using Real LightRAG API
INFO: ======================================================================
INFO: Evaluation Models:
INFO:   • LLM Model:            gpt-4o-mini
INFO:   • Embedding Model:      text-embedding-3-large
INFO:   • LLM Endpoint:         https://api.openai.com/v1
INFO:   • Bypass N-Parameter:   Enabled (use LangchainLLMWrapper for compatibility)
INFO: Concurrency & Rate Limiting:
INFO:   • Query Top-K:          10 Entities/Relations
INFO:   • LLM Max Retries:      5
INFO:   • LLM Timeout:          360 seconds
INFO: Test Configuration:
INFO:   • Total Test Cases:     6
INFO:   • Test Dataset:         sample_dataset.json
INFO:   • LightRAG API:         http://localhost:9621
INFO:   • Results Directory:    results
INFO: ======================================================================
INFO: 🚀 Starting RAGAS Evaluation of LightRAG System
INFO: 🔧 RAGAS Evaluation (Stage 2): 1 concurrent
INFO: ======================================================================
INFO:                                                                                                                                                      
INFO: ===================================================================================================================
INFO: 📊 EVALUATION RESULTS SUMMARY
INFO: ===================================================================================================================
INFO: #    | Question                                           |  Faith | AnswRel | CtxRec | CtxPrec |  RAGAS | Status
INFO: -------------------------------------------------------------------------------------------------------------------
INFO: 1    | How does LightRAG solve the hallucination probl... | 0.8571 |  0.9856 | 1.0000 |  0.2065 | 0.7623 |      ✓
INFO: 2    | What are the three main components required in ... | 0.8571 |  0.6003 | 1.0000 |  0.0000 | 0.6144 |      ✓
INFO: 3    | How does LightRAG's retrieval performance compa... | 0.6842 |  0.7357 | 1.0000 |  0.7907 | 0.8027 |      ✓
INFO: 4    | What vector databases does LightRAG support and... | 0.9200 |  0.9723 | 1.0000 |  0.3271 | 0.8049 |      ✓
INFO: 5    | What are the four key metrics for evaluating RA... | 0.8095 |  0.8910 | 0.5000 |  0.0000 | 0.5501 |      ✓
INFO: 6    | What are the core benefits of LightRAG and how ... | 0.7600 |  0.9689 | 1.0000 |  0.6102 | 0.8348 |      ✓
INFO: ===================================================================================================================
INFO: 
INFO: ======================================================================
INFO: 📊 EVALUATION COMPLETE
INFO: ======================================================================
INFO: Total Tests:    6
INFO: Successful:     6
INFO: Failed:         0
INFO: Success Rate:   100.00%
INFO: Elapsed Time:   418.56 seconds
INFO: Avg Time/Test:  69.76 seconds
INFO: 
INFO: ======================================================================
INFO: 📈 BENCHMARK RESULTS (Average)
INFO: ======================================================================
INFO: Average Faithfulness:      0.8147
INFO: Average Answer Relevance:  0.8590
INFO: Average Context Recall:    0.9167
INFO: Average Context Precision: 0.3224
INFO: Average RAGAS Score:       0.7282
INFO: ----------------------------------------------------------------------
INFO: Min RAGAS Score:           0.5501
INFO: Max RAGAS Score:           0.8348
INFO: 
INFO: ======================================================================
INFO: 📁 GENERATED FILES
INFO: ======================================================================
INFO: Results Dir:    D:\workspace\LightRAG\lightrag\evaluation\results
INFO:    • CSV:  results_20260406_221209.csv
INFO:    • JSON: results_20260406_221209.json
INFO: ======================================================================