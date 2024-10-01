FROM public.ecr.aws/lambda/python:3.8

RUN pip install requests
RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl


COPY clothing-model.tflite .
COPY inference_tflite.py .

CMD ["inference_tflite.lambda_handler"]
