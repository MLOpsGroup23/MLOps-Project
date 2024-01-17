from hydra import initialize, compose
from MLOps_Project.models.resnet import ResNet34
from MLOps_Project.visualizations.visualize_data import make_data_visualization
from MLOps_Project.visualizations.visualize import Create_TSNE_image_from_model
from os.path import exists


class TestVisualization:
    def test_visualization(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config")
            make_data_visualization(cfg)

            assert exists(
                "./reports/figures/FashionMNIST_Dataset_samples.pdf"
            ), "Expected a visualization of the dataset samples as PDF to be made"
            assert exists(
                "./reports/figures/FashionMNIST_Dataset_samples.png"
            ), "Expected a visualization of the dataset samples as PNG to be made"

            assert exists(
                "./reports/figures/FashionMNIST_Dataset_class_distribution.pdf"
            ), "Expected a visualization of the class distribution as PDF to be made"
            assert exists(
                "./reports/figures/FashionMNIST_Dataset_class_distribution.png"
            ), "Expected a visualization of the class distribution as PNG to be made"

            assert exists(
                "./reports/figures/FashionMNIST_Dataset_class_distribution_test.pdf"
            ), "Expected a visualization of the testing dataset samples as PDF to be made"
            assert exists(
                "./reports/figures/FashionMNIST_Dataset_class_distribution_test.png"
            ), "Expected a visualization of the testing dataset samples as PNG to be made"

            assert exists(
                "./reports/figures/FashionMNIST_Dataset_class_distribution_val.pdf"
            ), "Expected a visualization of the validation dataset samples as PDF to be made"
            assert exists(
                "./reports/figures/FashionMNIST_Dataset_class_distribution_val.png"
            ), "Expected a visualization of the validation dataset samples as PNG to be made"

    def test_tsne(self):
        model = ResNet34(lr=0.003, dropout_rate=0.2, required_channels=3)  # Random untrained model
        Create_TSNE_image_from_model(model, "outputs", "testTSNE")
        assert exists(
            "outputs/testTSNE.png"
        ), "Expected the TSNE to be stored in a PNG file, but was not actually saved."
