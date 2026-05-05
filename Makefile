ingest:
	cd backend && python ingest.py

serve:
	cd backend && uvicorn app:app --reload

frontend:
	cd frontend && python -m http.server 5500

eval:
	cd backend && python eval_recall.py