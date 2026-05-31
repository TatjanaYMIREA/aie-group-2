import click
import json
from .predict import load_model, predict

@click.group()
def cli():
    """Happiness Predictor CLI"""
    pass

@cli.command()
@click.option("--input", "-i", required=True, help="JSON файл с данными")
@click.option("--model-path", default="./artifacts/happiness_model.pkl", help="Путь к модели")
def predict_cmd(input, model_path):
    """Предсказать индекс счастья"""
    load_model(model_path)
    with open(input, 'r') as f:
        data = json.load(f)
    result = predict(data)
    click.echo(f"Предсказанный индекс счастья: {result:.3f}")

@cli.command()
def health():
    """Проверка работоспособности"""
    click.echo("✅ Сервис готов")

if __name__ == "__main__":
    cli()