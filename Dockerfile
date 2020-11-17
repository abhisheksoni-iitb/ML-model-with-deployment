FROM continuumio/anaconda:4.0.0
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python app.py