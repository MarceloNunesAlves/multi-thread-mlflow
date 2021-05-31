from concurrent.futures import ThreadPoolExecutor
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import os, getpass
import time
import mlflow
import random
import sys
import logging

runs = range(10)

logger = logging.getLogger(__name__)

filename = str(__file__)

logger.setLevel(logging.INFO)

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger.addHandler(console_handler)

def do_something_in_parallel():
    # do something
    name_exp = 'experiment_' + str(random.randint(1, 1000))

    logger.info('Inicio do processo')

    new_experiment = mlflow.create_experiment(name_exp)
    experiment = mlflow.get_experiment_by_name(name=name_exp)

    try:
        tags = {'mlflow.user': getpass.getuser(),
                'mlflow.source.name': filename,
                'mlflow.source.type': 'LOCAL'}

        # create a run
        run = client.create_run(experiment.experiment_id, tags=tags)

        logger.info("status: {}".format(run.info.status))

        # log parameter to this specific run
        learning_rate = random.uniform(0, 1)
        client.log_param(run.info.run_id, "learning_rate", learning_rate)
        logger.info("{} => parametro: learning_rate - valor: {}".format(name_exp, learning_rate))
        time.sleep(random.randint(6, 12))
        other_param = random.uniform(0, 1)
        client.log_param(run.info.run_id, "other_param", other_param)
        logger.info("{} => parametro: other_param - valor: {}".format(name_exp, other_param))
        time.sleep(random.randint(7, 13))

        client.set_terminated(run.info.run_id, 'FINISHED')

    except Exception as e:
        logger.error('{} => Erro no modelo. Motivo: {}'.format(name_exp, str(e)))
        client.set_tag(run.info.run_id, "mlflow.note.content",
                               "Erro na executação do modelo, motivo: {}"
                               .format(str(e)))
        client.set_terminated(run.info.run_id, 'FAILED')

    logger.info('Fim do processo')

def do_something_in_parallel_normal():
    logger.info('Inicio do processo')

    # do something
    name_exp = 'experiment_' + str(random.randint(1, 1000))

    new_experiment = mlflow.create_experiment(name_exp)
    experiment = mlflow.get_experiment_by_name(name=name_exp)

    try:
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            try:

                # log parameter to this specific run
                learning_rate = random.uniform(0, 1)
                mlflow.log_param("learning_rate", learning_rate)
                logger.info("{} => parametro: learning_rate - valor: {}".format(name_exp, learning_rate))
                time.sleep(random.randint(6, 12))
                other_param = random.uniform(0, 1)
                mlflow.log_param("other_param", other_param)
                logger.info("{} => parametro: other_param - valor: {}".format(name_exp, other_param))
                time.sleep(random.randint(7, 13))

            except Exception as e:
                run = mlflow.active_run()
                logger.error('{} => Erro no modelo. Motivo: {}'.format(name_exp, str(e)))
                client.set_tag(run.info.run_id, "mlflow.note.content",
                                       "Erro na executação do modelo, motivo: {}"
                                       .format(str(e)))
                raise e
    except Exception as ex:
        logger.error('Erro no modelo. Motivo: {}'.format(str(ex)))

    logger.info('Fim do processo')

if __name__ == '__main__':
    mlflow.set_tracking_uri(os.environ.get('URI_MLFLOW', 'http://localhost:5000'))

    client = MlflowClient()

    count = 1
    executor = ThreadPoolExecutor(max_workers=3)
    for x in runs:
        executor.submit(do_something_in_parallel)
        logger.info('Item enviado -> {}'.format(count))
        count += 1