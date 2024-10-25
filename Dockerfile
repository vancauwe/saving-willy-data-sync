FROM nvcr.io/nvidia/pytorch:22.02-py3

COPY . .
RUN pip install -r requirements.txt

ENTRYPOINT bash
