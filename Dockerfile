FROM tiangolo/uvicorn-gunicorn-machine-learning:python3.7
WORKDIR /app
COPY . /app

RUN conda install -c conda-forge flask-restful
RUN conda install -c conda-forge pandas
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge keras

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]