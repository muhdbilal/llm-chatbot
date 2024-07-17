Basic LLM chatbot with retrieval augmented generation (RAG), built using Langchain and Chainlit

* To run the basic version of the chatbot, run the following:

```
chainlit run chatbot.py
```

* To run the version of the chatbot with RAG, run the following:

```
chainlit run chatbot_with_pdf.py
```

The basic version of the chatbot will ask for your Hugginface token before starting. The RAG verson of the chatbot will also ask you for the pdf file it is supposed to draw data from.