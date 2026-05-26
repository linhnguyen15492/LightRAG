# RAGAS Evaluation Results - LightRAG System (Real API)

```bash
INFO: ======================================================================
INFO: 🔍 RAGAS Evaluation - Using Real LightRAG API
INFO: ======================================================================
WARNING: Skipping test case at index 26: missing question/query or ground_truth/result
WARNING: Skipping test case at index 28: missing question/query or ground_truth/result
WARNING: Skipping test case at index 82: missing question/query or ground_truth/result
WARNING: Skipping test case at index 90: missing question/query or ground_truth/result
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
INFO:   • Total Test Cases:     121
INFO:   • Test Dataset:         result.json
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
INFO: 1    | How do the various thematic elements across the... | 0.0000 |  0.5773 | 0.5714 |  0.0000 | 0.2872 |      ✓
INFO: 2    | What narrative techniques are employed by diffe... | 0.0000 |  0.9752 | 0.0000 |  0.0000 | 0.2438 |      ✓
INFO: 3    | How does each passage's historical context infl... | 0.1111 |  0.9413 | 0.0000 |  0.0000 | 0.2631 |      ✓
INFO: 4    | In what ways do the language and style evolve a... | 0.0000 |  0.9186 | 0.0000 |  0.0000 | 0.2297 |      ✓
INFO: 5    | How do the cultural references in each passage ... | 0.0000 |  0.9433 | 0.0000 |  0.0000 | 0.2358 |      ✓
INFO: 6    | What similarities and differences can be observ... | 0.0000 |  0.9725 | 0.0000 |  0.0000 | 0.2431 |      ✓
INFO: 7    | How do the authorsâ€™ backgrounds influence the... | 0.0000 |  0.9868 | 0.0000 |  0.0000 | 0.2467 |      ✓
INFO: 8    | What recurring motifs can be identified across ... | 0.0000 |  0.8733 | 0.6667 |  0.0000 | 0.3850 |      ✓
INFO: 9    | How does each authorâ€™s cultural background im... | 0.0000 |  0.8942 | 0.0000 |  0.0000 | 0.2236 |      ✓
INFO: 10   | What influence do historical events have on the... | 0.0000 |  0.9788 | 0.0000 |  0.0000 | 0.2447 |      ✓
INFO: 11   | How do the authors interpret the impact of majo... | 0.8000 |  0.9439 | 0.0000 |  0.0000 | 0.4360 |      ✓
INFO: 12   | What role does personal experience play in shap... | 0.0000 |  0.8654 | 0.0000 |  0.0000 | 0.2163 |      ✓
INFO: 13   | How are collective national memories reflected ... | 0.0000 |  0.8353 | 0.2500 |  0.0000 | 0.2713 |      ✓
INFO: 14   | In what ways do the events of the times affect ... | 0.0000 |  0.8073 | 0.0000 |  0.0000 | 0.2018 |      ✓
INFO: 15   | How does understanding the historical context e... | 0.0000 |  0.9601 | 0.0000 |  0.0000 | 0.2400 |      ✓
INFO: 16   | How do the portrayals of female characters diff... | 0.0000 |  0.8759 | 0.0000 |  0.0000 | 0.2190 |      ✓
INFO: 17   | What commentary do the texts provide on the soc... | 0.0000 |  0.7806 | 0.0000 |  0.0000 | 0.1952 |      ✓
INFO: 18   | How are the relationships between male and fema... | 0.7619 |  0.8524 | 0.1111 |  0.0000 | 0.4313 |      ✓
INFO: 19   | In what ways does the dataset reveal shifts in ... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 20   | How do the authors challenge or reinforce tradi... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 21   | How does an author's use of literary devices af... | 0.0000 |  0.9581 | 0.0000 |  0.0000 | 0.2395 |      ✓
INFO: 22   | In what ways does the complexity of language ch... | 0.0000 |  0.8678 | 0.0000 |  0.0000 | 0.2170 |      ✓
INFO: 23   | How can word choice and sentence structure infl... | 1.0000 |  0.9623 | 1.0000 |  0.0000 | 0.7406 |      ✓
INFO: 24   | How do style and form shape the themes and mess... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 25   | What role does voice play in guiding reader eng... | 0.0000 |  0.8291 | 0.0000 |  0.0000 | 0.2073 |      ✓
INFO: 26   | What pedagogical strategies can be employed to ... | 1.0000 |  0.9264 | 0.0000 |  0.0000 | 0.4816 |      ✓
INFO: 27   | How can the study of gender representation in t... | 0.0000 |  0.9762 | 1.0000 |  0.0000 | 0.4940 |      ✓
INFO: 28   | What assessment methods should be designed to e... | 0.0000 |  0.8867 | 0.0000 |  0.0000 | 0.2217 |      ✓
INFO: 29   | How can students be encouraged to engage with t... | 1.0000 |  0.8908 | 1.0000 |  0.0000 | 0.7227 |      ✓
INFO: 30   | What activities might promote collaborative ana... | 1.0000 |  0.9783 | 0.0000 |  0.0000 | 0.4946 |      ✓
INFO: 31   | In what ways can technology enhance the explora... | 0.0000 |  0.9101 | 0.0000 |  0.0000 | 0.2275 |      ✓
INFO: 32   | How can role-playing exercises help students un... | 0.0000 |  0.7466 | 0.0000 |  0.0000 | 0.1867 |      ✓
INFO: 33   | What discussion prompts can facilitate deep and... | 0.0000 |  0.8841 | 0.0000 |  0.0000 | 0.2210 |      ✓
INFO: 34   | How does adapting a literary work into film alt... | 0.0000 |  0.0000 | 1.0000 |  0.0000 | 0.2500 |      ✓
INFO: 35   | What elements must be preserved to maintain the... | 0.0000 |  0.9316 | 0.0000 |  0.0000 | 0.2329 |      ✓
INFO: 36   | How do visual and auditory elements of film enh... | 0.6250 |  0.0000 | 0.0000 |  0.0000 | 0.1562 |      ✓
INFO: 37   | In what ways can students analyze adaptations t... | 0.0000 |  0.0000 | 1.0000 |  0.0000 | 0.2500 |      ✓
INFO: 38   | What criteria can be used to assess the effecti... | 1.0000 |  0.8001 | 1.0000 |  0.0000 | 0.7000 |      ✓
INFO: 39   | How do metaphors and similes in the passages co... | 0.0000 |  1.0000 | 0.0000 |  0.0000 | 0.2500 |      ✓
INFO: 40   | What role does symbolism play in connecting dis... | 0.0000 |  0.7538 | 0.0000 |  0.0000 | 0.1884 |      ✓
INFO: 41   | In what ways can the interpretation of a text c... | 0.0000 |  0.8401 | 0.0000 |  0.0000 | 0.2100 |      ✓
INFO: 42   | How can students be taught to identify and anal... | 0.9583 |  0.8324 | 0.0000 |  0.0000 | 0.4477 |      ✓
INFO: 43   | What literary devices appear to have the most p... | 1.0000 |  0.6916 | 0.0000 |  0.0000 | 0.4229 |      ✓
INFO: 44   | How do the authors approach social issues such ... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 45   | In what ways can the relevance of these social ... | 0.0000 |  0.7563 | 1.0000 |  0.0000 | 0.4391 |      ✓
INFO: 46   | How can literature serve as a mirror reflecting... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 47   | What strategies can educators employ to encoura... | 1.0000 |  0.9402 | 0.6818 |  0.0000 | 0.6555 |      ✓
INFO: 48   | How do discussions of social issues within the ... | 0.0000 |  0.9548 | 0.0000 |  0.0000 | 0.2387 |      ✓
INFO: 49   | How did the societal norms of the time shape th... | 0.0000 |  0.6569 | 0.2857 |  0.0000 | 0.2357 |      ✓
INFO: 50   | What role did these texts play in reflecting or... | 0.8571 |  0.7005 | 0.2000 |  0.0000 | 0.4394 |      ✓
INFO: 51   | How do the narratives illustrate the cultural t... | 0.0000 |  0.9927 | 0.0000 |  0.0000 | 0.2482 |      ✓
INFO: 52   | In what ways can analyzing these texts enhance ... | 0.0000 |  0.9624 | 0.0000 |  0.0000 | 0.2406 |      ✓
INFO: 53   | How might the cultural contributions of the aut... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 54   | What overarching cultural themes recur across t... | 0.0000 |  0.8612 | 0.0000 |  0.0000 | 0.2153 |      ✓
INFO: 55   | How do the selected texts provide insight into ... | 0.0000 |  0.9475 | 0.0000 |  0.0000 | 0.2369 |      ✓
INFO: 56   | In what ways do the characteristics of the lite... | 0.0000 |  0.5081 | 0.0000 |  0.0000 | 0.1270 |      ✓
INFO: 57   | How does cultural exchange manifest in the them... | 0.0000 |  0.8524 | 0.0000 |  0.0000 | 0.2131 |      ✓
INFO: 58   | How can the legacy of these narratives influenc... | 1.0000 |  0.0000 | 1.0000 |  0.0000 | 0.5000 |      ✓
INFO: 59   | How do major historical events resonate within ... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 60   | What specific authors respond to their historic... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 61   | In what ways do the textsâ€™ narratives illumin... | 0.0000 |  0.9240 | 0.9091 |  0.0000 | 0.4583 |      ✓
INFO: 62   | How can readers engage with the texts to gain a... | 0.9167 |  0.9359 | 1.0000 |  0.0000 | 0.7131 |      ✓
INFO: 63   | What comparative frameworks might be useful to ... | 1.0000 |  0.9207 | 0.0000 |  0.0000 | 0.4802 |      ✓
INFO: 64   | How do the narratives express or challenge noti... | 0.1875 |  0.9181 | 0.5652 |  0.0000 | 0.4177 |      ✓
INFO: 65   | In what ways do the depicted identities reflect... | 0.0000 |  0.5873 | 0.0000 |  0.0000 | 0.1468 |      ✓
INFO: 66   | What narrative strategies do authors use to neg... | 0.0000 |  0.9035 | 0.0000 |  0.0000 | 0.2259 |      ✓
INFO: 67   | How can we view the authorsâ€™ explorations of ... | 0.0000 |  0.8719 | 0.0000 |  0.0000 | 0.2180 |      ✓
INFO: 68   | What insights about identity development can be... | 0.0000 |  0.8039 | 0.0000 |  0.0000 | 0.2010 |      ✓
INFO: 69   | How can we best convey the historical significa... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 70   | In what ways can visual elements enhance the st... | 0.0000 |  0.9848 | 0.0000 |  0.0000 | 0.2462 |      ✓
INFO: 71   | What methodologies can be used to blend histori... | 0.0000 |  0.9567 | 1.0000 |  0.0000 | 0.4892 |      ✓
INFO: 72   | How does framing historical events through lite... | 0.0000 |  0.8703 | 0.0000 |  0.0000 | 0.2176 |      ✓
INFO: 73   | What resources and tools are available to effec... | 0.0000 |  0.9551 | 0.2000 |  0.0000 | 0.2888 |      ✓
INFO: 74   | What criteria should be used to evaluate the li... | 0.0000 |  0.9035 | 0.2500 |  0.0000 | 0.2884 |      ✓
INFO: 75   | How might an authorâ€™s biography inform a crit... | 0.0000 |  0.9245 | 0.0000 |  0.0000 | 0.2311 |      ✓
INFO: 76   | In what ways do differing perspectives on theme... | 0.4375 |  0.8760 | 0.0000 |  0.0000 | 0.3284 |      ✓
INFO: 77   | How can interplay between the narrative style a... | 0.0000 |  0.8081 | 0.6842 |  0.0000 | 0.3731 |      ✓
INFO: 78   | What techniques should critics utilize in ensur... | 0.9286 |  0.9899 | 1.0000 |  0.0000 | 0.7296 |      ✓
INFO: 79   | How do characters evolve throughout the differe... | 0.0000 |  0.8217 | 0.0000 |  0.0000 | 0.2054 |      ✓
INFO: 80   | How can contrasting character arcs in the datas... | 0.0000 |  0.9872 | 0.0000 |  0.0000 | 0.2468 |      ✓
INFO: 81   | What role does dialogue play in character devel... | 0.0000 |  0.8800 | 0.0000 |  0.0000 | 0.2200 |      ✓
INFO: 82   | How does the setting contribute to the characte... | 0.0000 |  0.8166 | 0.0000 |  0.0000 | 0.2042 |      ✓
INFO: 83   | What literary devices are used to convey emotio... | 0.0000 |  0.9197 | 0.0000 |  0.0000 | 0.2299 |      ✓
INFO: 84   | In what ways do authors engage with themes of p... | 0.0000 |  0.8975 | 0.0000 |  0.0000 | 0.2244 |      ✓
INFO: 85   | How do emotional states in the texts interact w... | 0.0000 |  0.9436 | 0.0000 |  0.0000 | 0.2359 |      ✓
INFO: 86   | How can readers interpret psychological realism... | 0.0000 |  0.9206 | 1.0000 |  0.0000 | 0.4801 |      ✓
INFO: 87   | What recurring literary devices appear across t... | 0.0000 |  0.7031 | 0.0000 |  0.0000 | 0.1758 |      ✓
INFO: 88   | How do the identified devices enhance a readerâ... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 89   | In what ways might differing uses of the same l... | 1.0000 |  0.9781 | 1.0000 |  0.0000 | 0.7445 |      ✓
INFO: 90   | How does understanding an authorâ€™s unique sty... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 91   | What is the significance of genre in framing th... | 0.0000 |  0.7174 | 0.0000 |  0.0000 | 0.1794 |      ✓
INFO: 92   | How do the personal backgrounds of authors shap... | 0.2222 |  0.8785 | 1.0000 |  0.0000 | 0.5252 |      ✓
INFO: 93   | What significant influences from their cultural... | 0.0000 |  0.4287 | 0.0000 |  0.0000 | 0.1072 |      ✓
INFO: 94   | In what ways can an authorâ€™s life trajectory ... | 0.0000 |  0.9092 | 0.0000 |  0.0000 | 0.2273 |      ✓
INFO: 95   | How can comparative studies of multiple authors... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 96   | What interdisciplinary approaches might enhance... | 0.0000 |  0.9837 | 0.0000 |  0.0000 | 0.2459 |      ✓
INFO: 97   | How might thematic coherence across selected pa... | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 |      ✓
INFO: 98   | What considerations should be made when present... | 1.0000 |  0.9450 | 1.0000 |  0.0000 | 0.7363 |      ✓
INFO: 99   | How can the cultural background and historical ... | 0.0000 |  0.0000 | 1.0000 |  0.0000 | 0.2500 |      ✓
INFO: 100  | What marketing strategies can effectively commu... | 1.0000 |  0.8398 | 0.0000 |  0.0000 | 0.4600 |      ✓
INFO: 101  | How could visual elements in the book design en... | 0.0000 |  0.9268 | 0.0000 |  0.0000 | 0.2317 |      ✓
INFO: 102  | What messaging strategies can resonate with the... | 0.0000 |  0.3872 | 0.0000 |  0.0000 | 0.0968 |      ✓
INFO: 103  | How can social media platforms be effectively u... | 0.0000 |  0.9322 | 0.0000 |  0.0000 | 0.2331 |      ✓
INFO: 104  | In what ways could partnerships with literary b... | 0.0000 |  0.9732 | 0.0000 |  0.0000 | 0.2433 |      ✓
INFO: 105  | What formats (e.g., book trailers, interviews, ... | 0.0000 |  0.7704 | 0.0000 |  0.0000 | 0.1926 |      ✓
INFO: 106  | How can seasonal trends in literature influence... | 0.0000 |  0.9601 | 0.0000 |  0.0000 | 0.2400 |      ✓
INFO: 107  | What best practices can be derived from analyzi... | 0.0000 |  0.8945 | 0.0000 |  0.0000 | 0.2236 |      ✓
INFO: 108  | How can editorial feedback be tailored to highl... | 1.0000 |  0.9236 | 0.0000 |  0.0000 | 0.4809 |      ✓
INFO: 109  | How might editors balance creative freedom with... | 0.0417 |  0.9313 | 0.0000 |  0.0000 | 0.2432 |      ✓
INFO: 110  | What metrics should be utilized to evaluate the... | 0.0000 |  1.0000 | 0.7059 |  0.0000 | 0.4265 |      ✓
INFO: 111  | How can the dynamics of the publishing industry... | 1.0000 |  0.9927 | 1.0000 |  0.0000 | 0.7482 |      ✓
INFO: 112  | What criteria can be used to evaluate the marke... | 0.0000 |  0.9133 | 0.0000 |  0.0000 | 0.2283 |      ✓
INFO: 113  | How does understanding literary trends inform d... | 0.0000 |  0.8273 | 1.0000 |  0.0000 | 0.4568 |      ✓
INFO: 114  | In what ways does audience feedback shape the p... | 0.0000 |  0.8661 | 0.0000 |  0.0000 | 0.2165 |      ✓
INFO: 115  | How do collaborations with literary festivals o... | 1.0000 |  0.9889 | 0.0000 |  0.0000 | 0.4972 |      ✓
INFO: 116  | What data-driven approaches can be employed to ... | 0.0000 |  0.9685 | 0.0000 |  0.0000 | 0.2421 |      ✓
INFO: 117  | How can workshops be structured to foster meani... | 0.0000 |  0.9088 | 0.0000 |  0.0000 | 0.2272 |      ✓
INFO: 118  | What formats (panels, readings, Q&A) can be uti... | 1.0000 |  0.8284 | 0.0000 |  0.0000 | 0.4571 |      ✓
INFO: 119  | How can community outreach around the workshops... | 1.0000 |  0.0000 | 0.3333 |  0.0000 | 0.3333 |      ✓
INFO: 120  | What role does feedback from workshop participa... | 0.0000 |  0.9658 | 0.0000 |  0.0000 | 0.2415 |      ✓
INFO: 121  | How can event recordings be repurposed to expan... | 1.0000 |  0.9887 | 0.0000 |  0.0000 | 0.4972 |      ✓
INFO: ===================================================================================================================
INFO: 
INFO: ======================================================================
INFO: 📊 EVALUATION COMPLETE
INFO: ======================================================================
INFO: Total Tests:    121
INFO: Successful:     121
INFO: Failed:         0
INFO: Success Rate:   100.00%
INFO: Elapsed Time:   5227.23 seconds
INFO: Avg Time/Test:  43.20 seconds
INFO: 
INFO: ======================================================================
INFO: 📈 BENCHMARK RESULTS (Average)
INFO: ======================================================================
INFO: Average Faithfulness:      0.2054
INFO: Average Answer Relevance:  0.7391
INFO: Average Context Recall:    0.2018
INFO: Average Context Precision: 0.0000
INFO: Average RAGAS Score:       0.2866
INFO: ----------------------------------------------------------------------
INFO: Min RAGAS Score:           0.0000
INFO: Max RAGAS Score:           0.7482
INFO: 
INFO: ======================================================================
INFO: 📁 GENERATED FILES
INFO: ======================================================================
INFO: Results Dir:    D:\workspace\LightRAG\lightrag\evaluation\results
INFO:    • CSV:  results_20260524_191912.csv
INFO:    • JSON: results_20260524_191912.json
INFO: ======================================================================
```
