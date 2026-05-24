# RAGAS Evaluation Results - LightRAG System (Real API)

```bash
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
INFO:   • Total Test Cases:     50
INFO:   • Test Dataset:         mix_dataset.json
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
INFO: 1    | What are the recurring themes found across diff... | 0.0000 |  0.8017 | 0.0000 |  0.0000 | 0.2004 |      ✓
INFO: 2    | How do the stylistic elements of the poems cont... | 0.0000 |  0.9701 | 0.0000 |  0.0000 | 0.2425 |      ✓
INFO: 3    | In what ways do the historical contexts mention... | 0.0000 |  0.7613 | 0.0000 |  0.0000 | 0.1903 |      ✓
INFO: 4    | How can the data be employed to compare themes ... | 0.0000 |  0.9590 | 0.0000 |  0.0000 | 0.2398 |      ✓
INFO: 5    | What methodologies can be adopted to quantitati... | 0.0000 |  0.5685 | 0.0000 |  0.0000 | 0.1421 |      ✓
INFO: 6    | What literary devices are most frequently utili... | 0.0000 |  0.9741 | 0.3333 |  0.0000 | 0.3269 |      ✓
INFO: 7    | How do meter and rhyme schemes vary across diff... | 0.7778 |  0.0000 | 0.2727 |  0.0000 | 0.2626 |      ✓
INFO: 8    | In what ways can a close textual analysis of th... | 0.0000 |  0.9874 | 0.0000 |  0.0000 | 0.2469 |      ✓
INFO: 9    | How does the use of imagery and symbolism evolv... | 0.0000 |  0.9008 | 0.0000 |  0.0000 | 0.2252 |      ✓
INFO: 10   | Can the dataset provide insights into the evolu... | 0.0000 |  0.9421 | 0.3333 |  0.0000 | 0.3189 |      ✓
INFO: 11   | What machine learning models are most effective... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 12   | How does sentiment vary among different poets a... | 0.0952 |  0.0000 | 0.0909 |  0.0000 | 0.0465 |      ✓
INFO: 13   | Can sentiment trends be observed relative to sp... | 0.2500 |  0.0000 | 1.0000 |  0.0000 | 0.3125 |      ✓
INFO: 14   | What preprocessing steps are essential to optim... | 0.0417 |  0.9690 | 0.0000 |  0.0000 | 0.2527 |      ✓
INFO: 15   | How can sentiment analysis results be visualize... | 1.0000 |  0.8443 | 0.6667 |  0.0000 | 0.6277 |      ✓
INFO: 16   | Which topic modeling techniques, such as LDA or... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 17   | How do the results of topic modeling compare be... | 0.5000 |  0.0000 | 0.1667 |  0.0000 | 0.1667 |      ✓
INFO: 18   | In what ways can topic modeling enhance our und... | 0.0000 |  0.9521 | 1.0000 |  0.0000 | 0.4880 |      ✓
INFO: 19   | How can keywords extracted from the topic model... | 0.0000 |  0.9670 | 0.0000 |  0.0000 | 0.2417 |      ✓
INFO: 20   | What are the limitations of topic modeling in c... | 0.0000 |  0.9033 | 0.0000 |  0.0000 | 0.2258 |      ✓
INFO: 21   | How can themes from the poetry dataset be integ... | 0.9200 |  0.0000 | 0.9091 |  0.0000 | 0.4573 |      ✓
INFO: 22   | What specific selections from the dataset are m... | 0.0000 |  0.5386 | 0.1429 |  0.0000 | 0.1704 |      ✓
INFO: 23   | How might teaching strategies differ when using... | 1.0000 |  0.8656 | 0.0000 |  0.0000 | 0.4664 |      ✓
INFO: 24   | In what ways can the dataset be used to encoura... | 0.0000 |  0.9108 | 1.0000 |  0.0000 | 0.4777 |      ✓
INFO: 25   | How can students be assessed on their understan... | 1.0000 |  0.7764 | 0.0000 |  0.0000 | 0.4441 |      ✓
INFO: 26   | How can the literary techniques observed in the... | 0.0435 |  0.9867 | 0.6667 |  0.0000 | 0.4242 |      ✓
INFO: 27   | What types of prompts can be developed from the... | 0.0000 |  0.8808 | 0.0000 |  0.0000 | 0.2202 |      ✓
INFO: 28   | How can students be encouraged to analyze their... | 1.0000 |  0.9312 | 0.9231 |  0.0000 | 0.7136 |      ✓
INFO: 29   | How might peer review sessions be structured to... | 1.0000 |  0.9301 | 0.9231 |  0.0000 | 0.7133 |      ✓
INFO: 30   | What additional resources (books, articles, onl... | 0.9091 |  0.9098 | 0.0000 |  0.0000 | 0.4547 |      ✓
INFO: 31   | What distinctive features can be identified whe... | 0.0000 |  0.7739 | 0.2000 |  0.0000 | 0.2435 |      ✓
INFO: 32   | How do the poets' cultural backgrounds influenc... | 0.0000 |  0.8695 | 0.6000 |  0.0000 | 0.3674 |      ✓
INFO: 33   | What historical or biographical context is esse... | 0.0000 |  0.7278 | 0.0000 |  0.0000 | 0.1820 |      ✓
INFO: 34   | What ethical considerations should be taken int... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 35   | How can the dataset facilitate interdisciplinar... | 0.0000 |  0.9183 | 0.0000 |  0.0000 | 0.2296 |      ✓
INFO: 36   | How can the themes identified in the dataset be... | 0.0000 |  0.0000 | 0.9091 |  0.0000 | 0.2273 |      ✓
INFO: 37   | What poets from the dataset should be highlight... | 0.0000 |  0.0000 | 0.2000 |  0.0000 | 0.0500 |      ✓
INFO: 38   | How can participants be encouraged to engage wi... | 0.8095 |  0.8518 | 0.6250 |  0.0000 | 0.5716 |      ✓
INFO: 39   | What formats (panel discussions, workshops) wou... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 40   | How can the poetry readings be documented effec... | 1.0000 |  0.8872 | 1.0000 |  0.0000 | 0.7218 |      ✓
INFO: 41   | What criteria should guide the selection proces... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 42   | How does the thematic diversity in the dataset ... | 0.0000 |  0.9146 | 0.0000 |  0.0000 | 0.2287 |      ✓
INFO: 43   | In what ways can the anthology reflect the hist... | 0.0000 |  0.9649 | 0.0000 |  0.0000 | 0.2412 |      ✓
INFO: 44   | What supplementary materials (essays, author no... | 0.1429 |  0.0000 | 0.0000 |  0.0000 | 0.0357 |      ✓
INFO: 45   | How inclusive should the selection process be t... | 0.0000 |  0.0000 | 0.6667 |  0.0000 | 0.1667 |      ✓
INFO: 46   | What metrics can be utilized to assess reader e... | 0.0000 |  0.9565 | 0.0000 |  0.0000 | 0.2391 |      ✓
INFO: 47   | How can feedback from early readers or focus gr... | 0.0000 |  0.0000 | 1.0000 |  0.0000 | 0.2500 |      ✓
INFO: 48   | In what ways could social media platforms enhan... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 49   | How should the success of the anthology be defi... | 0.0000 |  0.8295 | 0.0000 |  0.0000 | 0.2074 |      ✓
INFO: 50   | What strategies can be implemented to promote d... | 1.0000 |  0.0000 | 1.0000 |  0.0000 | 0.5000 |      ✓
INFO: ===================================================================================================================
INFO:
INFO: ======================================================================
INFO: 📊 EVALUATION COMPLETE
INFO: ======================================================================
INFO: Total Tests:    50
INFO: Successful:     50
INFO: Failed:         0
INFO: Success Rate:   100.00%
INFO: Elapsed Time:   1101.44 seconds
INFO: Avg Time/Test:  22.03 seconds
INFO:
INFO: ======================================================================
INFO: 📈 BENCHMARK RESULTS (Average)
INFO: ======================================================================
INFO: Average Faithfulness:      0.2298
INFO: Average Answer Relevance:  0.5785
INFO: Average Context Recall:    0.2926
INFO: Average Context Precision: 0.0000
INFO: Average RAGAS Score:       0.2752
INFO: ----------------------------------------------------------------------
INFO: Min RAGAS Score:           0.0000
INFO: Max RAGAS Score:           0.7218
INFO:
INFO: ======================================================================
INFO: 📁 GENERATED FILES
INFO: ======================================================================
INFO: Results Dir:    D:\workspace\LightRAG\lightrag\evaluation\results
INFO:    • CSV:  results_20260524_134732.csv
INFO:    • JSON: results_20260524_134732.json
INFO: ======================================================================
```
