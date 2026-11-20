"""Dataset loader providing images, IDs, and labels for individual shark identification."""

import os

import torch
import torch.utils.data
import torchvision.io
import torchvision.transforms
import torchvision.transforms.v2

from src.core import constants
from src.core import utils
from src.retrieval import generation_store


class IndividualDataset(torch.utils.data.Dataset):
    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, broad-exception-caught
    """PyTorch Dataset loading shark images, IDs, and labels for model training and evaluation."""

    def __init__(self,
                 images_path,
                 annotations_path,
                 train_annotations_path=None,
                 test_annotations_path=None,
                 channels=3,
                 output_image_size=None,
                 normalization=None,
                 device=torch.device('cpu'),
                 csv_dir=None):
        """Initialize the dataset with paths, preprocessing configuration, and optional CSV metadata directory."""
        self.dataset = self
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.csv_dir = csv_dir
        self.channels = channels
        self.output_image_size = output_image_size
        self.normalization = normalization
        # Extract resize mode from normalization config if present, or default to pad
        self.resize_mode = "pad"
        if self.normalization and "resize_mode" in self.normalization:
            self.resize_mode = self.normalization["resize_mode"]

        self.device = device
        self.images = {}

        if self.normalization:
            self.normalize_op = torchvision.transforms.v2.Normalize(
                mean=self.normalization['mean'],
                std=self.normalization['std']
            )
        else:
            self.normalize_op = None

        print(f"Loading individual dataset, {channels} channels. "
              f"Output size: {self.output_image_size}, Normalization: {self.normalization is not None}")

        # Lists to contain needed data.
        self.ids = []
        self.filenames = []
        self.labels = []

        # Cache train and test IDs directly from annotations so splits persist when saved to PT format.
        if train_annotations_path is None:
            # Fallback: look for sister train-annotations.tsv in the same directory as annotations_path.
            train_annotations_path = os.path.join(os.path.dirname(annotations_path), constants.TRAIN_ANNOTATIONS)

        if test_annotations_path is None:
            # Fallback: look for sister valid-annotations.tsv in the same directory as annotations_path.
            test_annotations_path = os.path.join(os.path.dirname(annotations_path), constants.VALID_ANNOTATIONS)

        # Load split definitions if files exist.
        train_annotations = []
        if os.path.exists(train_annotations_path):
            train_annotations = utils.load_csv_file(train_annotations_path, delimiter='\t')

        test_annotations = []
        if os.path.exists(test_annotations_path):
            test_annotations = utils.load_csv_file(test_annotations_path, delimiter='\t')

        self.train_ids = []
        self.test_ids = []
        for annotation in train_annotations:
            if not annotation:
                continue
            self.train_ids.append(int(annotation[0]))
        for annotation in test_annotations:
            if not annotation:
                continue
            self.test_ids.append(int(annotation[0]))

        # Load the ids to be returned in this dataset from the selected annotations file, train, test, full.
        annotations = utils.load_csv_file(annotations_path, delimiter='\t')
        for annotation in annotations:
            if not annotation:
                continue
            if not os.path.exists(os.path.join(self._get_resolved_images_path(), annotation[1])):
                print(f"Image {annotation[1]} does not exist.")
                continue
            self.ids.append(int(annotation[0]))
            self.filenames.append(annotation[1])
            self.labels.append(int(annotation[2]))

        print("Dataset size: ", len(self.ids))

        # Check ids and filenames lists are same length
        assert len(self.ids) == len(self.filenames) == len(self.labels)

        # Store labels as tensor.
        # Keep labels on CPU initially, they will be moved to device by the Trainer.
        self.labels = torch.tensor(self.labels, dtype=torch.long, device=torch.device('cpu'))

    def _read_image(self, filename):
        """Return the decoded RGB image tensor for a dataset filename, naming the file if it cannot be read."""
        image_path = os.path.join(self._get_resolved_images_path(), filename)
        try:
            return torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        except Exception as e_img:
            raise RuntimeError(f"Cannot read image {image_path}: {e_img}") from e_img

    def check_images_readable(self):
        """Decode every image once and raise a single error listing the unreadable ones."""
        print(f"Checking that all {len(self.filenames)} images can be read.")
        unreadable = []
        for filename in self.filenames:
            try:
                self._read_image(filename)
            except RuntimeError:
                unreadable.append(filename)
        if unreadable:
            raise RuntimeError(f"{len(unreadable)} unreadable image(s) in {self.annotations_path}: "
                               f"{', '.join(unreadable[:10])}{' ...' if len(unreadable) > 10 else ''}")

    def __getitem__(self, image_index):
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        """Return an image tensor and its integer label for the given index."""
        if not isinstance(image_index, int):
            raise TypeError(f"IndividualDataset.__getitem__ expects an integer index, but got {type(image_index)}. "
                            "If you are trying to get all items, use a DataLoader or specific methods like get_labels().")

        filename = self.filenames[image_index]
        label = self.labels[image_index]

        img_original = self._read_image(filename)
        image = self._process_image_data(img_original, filename)
        return image, label

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert paths to relative paths before pickling for portability.
        csv_dir = state.get('csv_dir')
        if csv_dir and os.path.isabs(csv_dir):
            portable_csv_dir = utils.portable_csv_dir(csv_dir, include_snapshots=True)
            if portable_csv_dir is not None:
                state['csv_dir'] = portable_csv_dir

        for key in ('images_path', 'annotations_path'):
            stored_path = state.get(key)
            if stored_path and os.path.isabs(stored_path):
                rel = utils.relative_to_data_root(stored_path, constants.BASE_DATA_DIR)
                if rel is not None:
                    state[key] = rel

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'dataset' in state and state['dataset'] is not None:
            self.dataset = self
        csv_dir = getattr(self, 'csv_dir', None)
        if csv_dir and not os.path.isabs(csv_dir):
            resolved = utils.resolve_relative_csv_dir(csv_dir)
            if resolved:
                self.csv_dir = resolved

    def get_resolved_csv_dir(self):
        """Return the resolved absolute CSV directory for this dataset."""
        return utils.resolve_relative_csv_dir(getattr(self, 'csv_dir', None))

    def image_height(self):
        """Return the target image height after resizing."""
        if self.output_image_size:
            return self.output_image_size[0]
        raise ValueError("output_image_size was not set on this dataset. Pass it at construction.")

    def image_width(self):
        """Return the target image width after resizing."""
        if self.output_image_size:
            return self.output_image_size[1]
        raise ValueError("output_image_size was not set on this dataset. Pass it at construction.")

    def get_ids(self):
        """Return the list of all image IDs in the dataset."""
        return self.ids

    def get_filenames(self):
        """Return the list of all filenames in the dataset."""
        return self.filenames

    def get_labels(self):
        """Return the tensor of all labels in the dataset."""
        return self.labels

    def get_class_counts(self):
        """Return a tensor of class counts indexed by label using bincount."""
        labels = self.labels.cpu()
        return torch.bincount(labels)

    def get_train_ids(self):
        """Return the list of IDs used for training."""
        return self.train_ids

    def get_test_ids(self):
        """Return the list of IDs used for testing."""
        return self.test_ids

    def get_filename_by_id(self, individual_id):
        """Retrieve the filename corresponding to the given ID."""
        if individual_id not in self.ids:
            print(f"Image with id {individual_id} not found.")
            return None
        return self.filenames[self.ids.index(individual_id)]

    def get_label_by_id(self, individual_id):
        """Retrieve the label corresponding to the given ID."""
        if individual_id not in self.ids:
            print(f"Image with id {individual_id} not found.")
            return None
        return self.labels[self.ids.index(individual_id)]

    def get_id_by_filename(self, filename):
        """Retrieve the ID corresponding to the given filename."""
        if filename not in self.filenames:
            print(f"Image with filename {filename} not found.")
            return None
        return self.ids[self.filenames.index(filename)]

    def get_label_by_filename(self, filename):
        """Retrieve the label corresponding to the given filename."""
        index = self.filenames.index(filename)
        return self.labels.tolist()[index]

    def get_ids_by_label(self, label):
        """Get all IDs for a particular label or individual ID."""
        if label not in self.labels.tolist():
            print(f"Label with label {label} not found.")
        ids = []
        for index, item in enumerate(self.labels.tolist()):
            if item == label:
                ids.append(self.ids[index])
        return ids

    def get_filenames_by_label(self, label):
        """Get all filenames for a particular label or individual ID."""
        if label not in self.labels.tolist():
            print(f"Label with label {label} not found.")
        filenames = []
        for index, item in enumerate(self.labels.tolist()):
            if item == label:
                filenames.append(self.filenames[index])
        return filenames

    def load_images(self):
        # pylint: disable=too-many-branches, too-many-statements
        """Load dataset images onto the configured device."""
        for index, filename in enumerate(self.filenames):
            final_data = self._process_image_data(self._read_image(filename), filename)
            self.images[index] = final_data.to(self.device)

    @classmethod
    def create_and_store(cls, model_name, channels=3, annotations_version=None,
                         output_image_size=None, normalization=None, output_dir=None,
                         csv_dir=None):
        """Store the dataset as a PT file alongside embeddings for model training and evaluation."""
        utils.print_function_banner("create_and_store",
                                    f"Creating and storing dataset for model: {model_name}.")

        device = utils.get_torch_device()

        if annotations_version is None:
            annotations_version = "annotations"

        print(f"Using annotations version: {annotations_version}")
        version_dir = os.path.join(constants.ANNOTATIONS_DIR, annotations_version)
        full_annotations_path = os.path.join(version_dir, constants.FULL_ANNOTATIONS)
        train_annotations_path, valid_annotations_path = utils.get_annotation_paths(annotations_version)

        # Check for config to get image size and normalization if not provided.
        config_path = os.path.join(constants.TRAINED_MODELS_DIR, model_name, "config.json")
        if os.path.exists(config_path) and (output_image_size is None or normalization is None):
            try:
                print(f"Loading config from {config_path}")
                config = utils.load_json_file(config_path)
                data_config = config.get("data", {})

                if output_image_size is None and "image_size" in data_config:
                    output_image_size = tuple(data_config["image_size"])
                    print(f"Using image size from config: {output_image_size}")

                if normalization is None and "normalization" in data_config:
                    normalization = data_config["normalization"]
                    print(f"Using normalization from config: {normalization}")
            except Exception as e:
                print(f"Error loading config for dataset parameters: {e}")

        print("Loading and saving dataset.")
        dataset = cls(
            constants.IMAGES_DIR,
            full_annotations_path,
            train_annotations_path=train_annotations_path,
            test_annotations_path=valid_annotations_path,
            device=device,
            channels=channels,
            output_image_size=output_image_size,
            normalization=normalization,
            csv_dir=csv_dir,
        )
        dataset.check_images_readable()

        save_dir = output_dir if output_dir else os.path.join(
            constants.EMBEDDINGS_DIR, model_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(dataset, os.path.join(save_dir, "stored_dataset.pt"))

    @staticmethod
    def print_stored_info(model_name):
        """Print train and test split counts from a stored dataset PT file."""
        device = utils.get_torch_device()

        print("Printing dataset.")
        embeddings_dir = generation_store.resolve_active_directory(model_name)
        dataset = torch.load(os.path.join(embeddings_dir, "stored_dataset.pt"),
                             map_location=device, weights_only=False)

        print("Train and test split:")
        train_ids = dataset.get_train_ids()
        test_ids = dataset.get_test_ids()
        print("Train Ids:", len(train_ids))
        print("\n")
        print("Test Ids:", len(test_ids), test_ids)

    def _get_resolved_images_path(self):
        """Return the resolved absolute images path, handling relative or stale paths."""
        path = getattr(self, 'images_path', None)
        if not path:
            return constants.IMAGES_DIR
        if os.path.exists(path):
            return path
        if not os.path.isabs(path):
            candidate = os.path.join(constants.BASE_DATA_DIR, path)
            if os.path.exists(candidate):
                return candidate
            return constants.IMAGES_DIR
        if not os.path.exists(path) and os.path.exists(constants.IMAGES_DIR):
            return constants.IMAGES_DIR
        return path

    def _calculate_resize_params(self, current_h, current_w):
        """Calculate resize parameters and return the resize transform operator."""
        target_h, target_w = self.image_height(), self.image_width()
        intermediate_size = None

        if self.resize_mode == "squash":
            resize_img_op = torchvision.transforms.v2.Resize(size=(target_h, target_w),
                                                             antialias=True)
        elif current_h > target_h or current_w > target_w:
            ratio = min(target_h / current_h, target_w / current_w)
            intermediate_size = (int(current_h * ratio), int(current_w * ratio))
            resize_img_op = torchvision.transforms.v2.Resize(size=intermediate_size,
                                                             antialias=True)
        else:
            resize_img_op = torchvision.transforms.v2.Identity()

        return resize_img_op, intermediate_size

    def _process_image_data(self, img_original, filename):
        """Apply transformations, cropping, and normalization to an image tensor."""
        target_h, target_w = self.image_height(), self.image_width()
        current_h, current_w = img_original.shape[1], img_original.shape[2]

        resize_img_op, _ = self._calculate_resize_params(current_h, current_w)
        crop_op = torchvision.transforms.v2.CenterCrop(size=(target_h, target_w))
        to_float_op = torchvision.transforms.v2.ToDtype(torch.float32, scale=True)

        img_t = resize_img_op(img_original)
        img_t = crop_op(img_t)
        img_final = to_float_op(img_t)

        if self.normalize_op:
            img_final = self.normalize_op(img_final)

        return img_final


# Class name that stored pickles may reference.
SharkfinDataset = IndividualDataset


def main():
    """Module entry point for ad-hoc dataset inspection and debugging."""


if __name__ == "__main__":
    main()
