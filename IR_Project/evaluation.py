''' to run:
python -m venv env          # create env
env\Scripts\activate        # activate (Windows)
pip install faiss-cpu nltk rouge-score numpy openai transformers sentence-transformers
replace api_key with actual api key: 
python evaluation.py
to exit environment: deactivate
'''

from query import query
import faiss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

api_key = "" 


benchmark_qna = {
    "What is a PhD student's responsibility after bill clearance for the National Institute Travel Grant?": "1. After attending the conference/symposium, the research scholar must share their experiences and acquired knowledge with the department through a formal presentation. The Head of the Department should issue a notice for this presentation, and a copy should be submitted to the AGSRD office along with travel-related documents for bill clearance. 2. Submit the Travel Allowance (TA) and Daily Allowance (DA) form within 15 days of the trip, including accurate travel and DA details and all original vouchers and bills. Enclosures should include the registration certificate, registration fee receipt, travel tickets, and seminar notice indicating shared knowledge from the event. 3. If travel documents are unavailable, the accounts office will use a standard distance-based calculation from the origin to the destination. 4. In case the visit is canceled or the grants remain unused, please let the office know in writing.",
    
    "What is the eligibility criteria for availing BITS Pilani International Travel Award?": "1. Full-time PhD students of the Pilani Campus, KK Birla Goa Campus, and Hyderabad Campus of BITS Pilani are eligible to submit an application for the BITS Pilani International Travel Award. 2. PhD students after completing two years from the date of admission are eligible for consideration for the International Travel Award. 3. PhD students enrolled in dual degree programs (Cotutelle programs) are not eligible for the travel award. 4. A PhD student can receive the International Travel Award only once during their PhD program.",
    
    "What is the maximum limit of the grant for the International Travel Award?": "INR 1.5 lakhs",
    
    "What is the maximum limit of the grant for the National Travel Award?": "INR 25,000",
    
    "What is the checklist for a PhD research proposal?": "A PhD proposal should follow this layout: a. General Formatting b. Proposed Topic of Research c. Objectives of Proposed Research d. Background of Proposed Research e. Methodology and Work Plan f. References",
    
    "What are the items which can be purchased under the contingency (with Institute PhD Fellowship)?": "The items which can be procured under the Institute PhD contingency grant (for the scholars getting Institute fellowship) related to the Ph.D research work are: 1) Books and journals 2) Stationary, calculator, laser pointer, laptop mouse & cooling pad and printer cartridges 3) Supplies of computer consumables like external storage device, battery for UPS/Computers, anti- virus and other software (add-on) 4) Travel for attending conference, workshop and Lab visiting and sample collecting and CSIR NET Exams 5) Registration Examination fee for participation in professional development programs (conference, workshop, school etc and CSIR-UGC NET) 6) Chemicals/Glassware 7) Photocopying/Typing/Binding/ charges 8) Spare parts replacement and minor repair of computers/Laptops. 9) Charges for recording spectra other experimental facility utilization charges 10) Other consumable materials required for experimental/theoretical studies 11) Data cards/ Data purchases.",
    
    "What are the guidelines for availing casual leaves by a PhD student?": "Casual leave is approved by the Supervisor. Casual leaves cannot be clubbed with any other leaves.",
    
    "What is the difference between casual leaves and special casual leaves?": "Special casual leave is permissible for attending conferences/ workshops/ symposia/ training programmes etc. as approved by DRC while a casual leave need not be approved by the DRC. Special casual leave should not be clubbed with on duty leave or casual leave while casual leave cannot be clubbed with any other type of leaves.",
    
    "When is an on-duty leave applicable?": "On duty leave is applicable when the student is expected to perform PhD project work at an off-campus site without a formal agreement",
    
    "How many members are present in the Departmental Research Committee (DRC)?": "The DRC consists of the Head of Department (HOD) as ex-officio member and Chairperson, and 2-15 faculty members, at the level of Assistant Professor or above, who are active in research. The size of DRC may depend on the number of faculty members in the Department.",
    
    "What is the term of DRC members?": "The term of the DRC members is 2 years.",
    
    "How many credits should a PhD candidate holding only a first degree be prescribed?": "Students holding first degree (B.Tech, B.E., M.A, M.Sc etc.) must be prescribed a minimum of six courses spread in two semesters with a minimum of 24 units.",
    
    "Is vertical transfer from B.Tech/M.Tech to PhD possible?": "Yes. Selection & recommendation of vertical transfer students should be done by the HOD/DRC, only after their source of fellowship/funding is identified and approved.",
    
    "How much should a student secure to pass the PhD qualifying examination?": "The rigor & quality of Ph.D qualifying examination is the responsibility of individual DRC's. The student has to secure a minimum of 50 percent in individual sub areas.",
    
    "What are the proposed sub-areas for PhD Qualifying examination for the CS&IS department?": "1. AI, Machine Learning & Data Mining 2. Computer Architecture, Embedded Systems & Robotics 3. Databases & Data Warehousing CS/IS 4. High Performance & Distributed Computing 5. Image Processing & Multimedia 6. Networking & Mobile Computing 7. Algorithms Theoretical Computer Science",
    
    "What steps should be taken in case of a change in supervisor of a PhD student?": "A fresh approval of the University level Doctoral Counseling Committee based at Pilani, is required for any changes in topic of research and/or supervisor. Candidate has to request through supervisor and DRC to Associate Dean AGSRD, who will forward the request to Dean, AGSRD for approval of DCC/Research Board.",
    
    "In what language should the PhD thesis be written?": "English",
    
    "For submitting an industry research proposal, what are the different budget heads?": "Equipment, Consumables, Contingency, Manpower, Travel, Overhead",
    
    "What are the 3 key funds along which the overhead costs are distributed?": "The Professional Development Fund (PDF), the Department Development Fund (DDF), and the Campus Development Fund (CDF).",
    
    "What are the allocation percentages for Professional Development Fund (PDF), the Department Development Fund (DDF), and the Campus Development Fund (CDF) given that the overhead is less than 10 percent of the total budget?": "40%, 20%, 40%",
    
    "What are the allocation percentages for Professional Development Fund (PDF), the Department Development Fund (DDF), and the Campus Development Fund (CDF) given that the overhead is more than 10 percent of the total budget?": "60%, 20%, 20%",
    
    "What activities does the professional development fund support?": "Membership and conferences, Educational Materials and Equipment, Hardware Purchases, Research Assistance, Local Hospitality, Short-Term Courses and Training, Travel Support, Experimental Work, Professional Development Programs.",
    
    "What are the consulting restrictions for consulting activities for faculty members?": "No consulting activity should: • Interfere with Faculty Obligations: Detract from the faculty's ability to fulfill BITS Pilani's obligations. • Restrict Academic and Research Activities: Restrict or limit the faculty's ability to pursue their academic and/or research activities at BITS Pilani due to confidentiality terms, limited access to intellectual property, or other constraints. • Impair Intellectual Property Rights: Give away rights or assign intellectual property that is already owned by and assigned to BITS Pilani. Faculty members should be vigilant about any provisions, such as confidentiality intellectual property (IP), or non-competition clauses, that might restrict their current or future research and academic activities as institute faculty members.",
    
    "What is the purpose of the Standard Operating Procedures (SOP) in GCIR?": "The SOP aims to provide a clear, structured framework for managing grants and industrial research activities at BITS Pilani, Hyderabad Campus.",
    
    "What are the key steps involved in submitting a new research proposal?": "Access the GCIR website, navigate to the proposal submission section, complete the submission form, upload required documents, obtain the endorsement certificate, and finalize the submission.",
    
    "What documents are required for submitting a consultancy proposal?": "A consultancy proposal draft, Note for Approval (NFA), and any additional required agreements such as MOUs or NDAs.",
    
    "How many working days are required for the issue of an endorsement certificate?": "2–3 working days.",
    
    "How many days per year can faculty members allocate for consultancy projects?": "Up to 52 working days per year.",
    
    "How many types of budget heads are there for industry-sponsored projects?": "There are at least six: equipment, consumables, contingency, manpower, travel, and overhead costs.",
    
    "When should PI submit the final proposal to a funding agency?": "After obtaining the endorsement certificate from the GCIR office.",
    
    "When can research scholars apply for an experience certificate or relieving letter?": "After completing their project tenure or upon resignation.",
    
    "When must travel approvals be obtained for using project funds?": "Before undertaking any travel, a request must be submitted to the GCIR office for approval.",
    
    "How is time commitment for consultancy projects regulated?": "Faculty members must ensure that consultancy work does not interfere with their academic responsibilities and is limited to one working day per week.",
    
    "How is overhead cost distributed among different funds?": "Overhead costs are allocated among the Professional Development Fund (PDF), Department Development Fund (DDF), and Campus Development Fund (CDF) based on a set percentage.",
    
    "How is a research scholar appointed after project approval?": "After receiving project approval, an advertisement is issued, followed by the shortlisting of candidates, conducting interviews, and final selection with an offer letter."
}

query_scores = []

for question, answer in benchmark_qna.items():
    print(f"Question: {question}")
    print(f"Expected Answer: {answer}")
    print("Model Response:")
    hypothesis_answer = query(question, api_key)
    expected_answer = answer

    model = SentenceTransformer('all-MiniLM-L6-v2')  # fast + good
    emb1 = model.encode(expected_answer, convert_to_tensor=True)
    emb2 = model.encode(hypothesis_answer, convert_to_tensor=True)

    score = util.pytorch_cos_sim(emb1, emb2)
    print("BERT SCORE using MiniLM Model  ", score.item())  # cosine similarity between 0 and 1

    # Just split by space (no nltk tokenizer)
    ref_tokens = [expected_answer.split()]
    cand_tokens = hypothesis_answer.split()
    smoothie = SmoothingFunction().method1  # simple smoothing
    score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)    
    query_scores.append(score)
    print(f"BLEU Score: {score:.4f}")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(expected_answer, hypothesis_answer)

    print("ROUGE-1:", scores['rouge1'].fmeasure)
    print("ROUGE-2:", scores['rouge2'].fmeasure)
    print("ROUGE-L:", scores['rougeL'].fmeasure)
    print("\n" + "="*50 + "\n")




