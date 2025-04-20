from model import HiringModel
from api_service import HiringAPI
import uvicorn

app = HiringAPI().get_app()

def main():
    model = HiringModel()
    model.predict_user_input()
    model.grid_search()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
