from tensorflow.keras.utils import Sequence
import numpy as np
import os
import json
import settings


def print_experiment_data_info(experiment_data):
    print("*" * 30)
    print("=" * 30)
    print("Dataset name:        ", experiment_data["name"])
    print("Dataset description: ", experiment_data["description"])
    print("Tensor image size:   ", experiment_data["tensorImageSize"])
    print("Dataset release:     ", experiment_data["release"])
    print("Dataset reference:   ", experiment_data["reference"])
    print("Dataset license:     ", experiment_data["licence"])
    print("=" * 30)
    print("*" * 30)


def load_experiment_data(json_filename):
    try:
        with open(json_filename, "r") as fp:
            return json.load(fp)
    except IOError:
        raise Exception(
            f"File {json_filename} doesn't exist. It should be part of the Decathlon directory"
        )


def split_file_indices(num_files, split, seed):
    np.random.seed(seed)
    indices = np.arange(num_files)
    np.random.shuffle(indices)

    train_indices = int(np.floor(num_files * split))
    train_split = indices[:train_indices]

    other_split = indices[train_indices:]
    other_indices = len(other_split) // 2
    validate_split = other_split[:other_indices]
    test_split = other_split[other_indices:]

    return train_split, validate_split, test_split


def get_file_paths(data_path, experiment_data, file_indices):
    return [
        os.path.join(data_path, experiment_data["training"][idx]["label"])
        for idx in file_indices
    ]


def get_decathlon_filelist(data_path, seed=209, split=0.85):
    json_filename = os.path.join(data_path, "dataset.json")
    experiment_data = load_experiment_data(json_filename)

    # print_experiment_data_info(experiment_data)

    train_list, validate_list, test_list = split_file_indices(
        experiment_data["numTraining"], split, seed
    )

    train_files = get_file_paths(data_path, experiment_data, train_list)
    validate_files = get_file_paths(data_path, experiment_data, validate_list)
    test_files = get_file_paths(data_path, experiment_data, test_list)

    print(f"Number of training files   = {len(train_files)}")
    print(f"Number of validation files = {len(validate_files)}")
    print(f"Number of testing files    = {len(test_files)}")

    return train_files, validate_files, test_files


def z_score_normalize(image):
    return (image - image.mean()) / image.std()


def preprocess_label(label):
    label[label > 0] = 1.0
    return label


def augment_data(img, msk):
    if np.random.rand() > 0.5:
        ax = np.random.choice([0, 1])

        img = np.flip(img, ax)
        msk = np.flip(msk, ax)

    if np.random.rand() > 0.5:
        rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

        img = np.rot90(img, rot, axes=(0, 1))  # Rotate axes 0 and 1
        msk = np.rot90(msk, rot, axes=(0, 1))  # Rotate axes 0 and 1

    return img, msk


class DatasetGenerator(Sequence):
    def __init__(
        self,
        filenames,
        batch_size=8,
        crop_dim=(240, 240),
        augment=False,
        seed=209,
        dim=4,
    ):
        import nibabel as nib

        self.filenames = filenames
        self.batch_size = batch_size
        self.augment = augment
        self.seed = seed
        self.dim = dim
        self.num_files = len(filenames)

        # Load the first image to determine slice dimension and number of slices
        img = np.array(nib.load(filenames[0]).dataobj)
        self.slice_dim = 2  # We'll assume z-dimension (slice) is last
        self.num_slices_per_scan = img.shape[self.slice_dim]

        # Adjust crop dimensions if set to -1
        self.crop_dim = (
            img.shape[0] if crop_dim[0] == -1 else crop_dim[0],
            img.shape[1] if crop_dim[1] == -1 else crop_dim[1],
        )

        self.ds = self.get_dataset()

    def crop_input(self, img, msk):
        def calculate_crop_start(image_size, crop_size, is_random, offset):
            start = (image_size - crop_size) // 2
            if offset > 0 and is_random:
                start = min(
                    start + np.random.choice(range(-offset, offset)),
                    (image_size - crop_size) // 2,
                )
            return start

        is_random = self.augment and np.random.rand() > 0.5
        slices = []

        for i in range(2):
            crop_size = self.crop_dim[i]
            image_size = img.shape[i]

            offset = int(np.floor((image_size - crop_size) // 2 * 0.20))
            start = calculate_crop_start(image_size, crop_size, is_random, offset)
            slices.append(slice(start, start + crop_size))

        return img[tuple(slices)], msk[tuple(slices)]

    def generate_batch_from_files(self):
        """
        Python generator which goes through a list of filenames to load.
        The files are 3D image (slice is dimension index 2 by default). However,
        we need to yield them as a batch of 2D slices. This generator
        keeps yielding a batch of 2D slices at a time until the 3D image is
        complete and then moves to the next 3D image in the filenames.
        An optional `randomize_slices` allows the user to randomize the 3D image
        slices after loading if desired.
        """
        import nibabel as nib

        def load_and_preprocess_image(label_filename):
            image_filename = label_filename.replace("labelsTr", "imagesTr")

            if self.dim == 4:
                image = np.array(nib.load(image_filename).dataobj)[
                    :, :, :, 0
                ]  # Just take FLAIR channel (channel 0)
            else:
                image = np.array(nib.load(image_filename).dataobj)
            image = z_score_normalize(image)

            label = np.array(nib.load(label_filename).dataobj)
            label = preprocess_label(label)
            return self.crop_input(image, label)

        def stack_images(image_stack, label_stack, image, label):
            if image_stack is None:
                return image, label
            image_stack = np.concatenate((image_stack, image), axis=self.slice_dim)
            label_stack = np.concatenate((label_stack, label), axis=self.slice_dim)
            return image_stack, label_stack

        def get_batch(image, label, idy, num_slices):
            if (idy + self.batch_size) < num_slices:
                return (
                    image[:, :, idy : idy + self.batch_size],
                    label[:, :, idy : idy + self.batch_size],
                )
            return image[:, :, -self.batch_size :], label[:, :, -self.batch_size :]

        np.random.seed(self.seed)
        idx, idy = 0, 0

        while True:
            NUM_QUEUED_IMAGES = (
                1 + self.batch_size // self.num_slices_per_scan
            )  # Get enough for full batch + 1
            img_stack, label_stack = None, None

            for _ in range(NUM_QUEUED_IMAGES):
                label_filename = self.filenames[idx]
                img, label = load_and_preprocess_image(label_filename)
                img_stack, label_stack = stack_images(
                    img_stack, label_stack, img, label
                )
                idx = (idx + 1) % len(self.filenames)
                if idx == 0:
                    np.random.shuffle(
                        self.filenames
                    )  # Shuffle the filenames for the next iteration

            img, label = img_stack, label_stack
            num_slices = img.shape[self.slice_dim]

            if self.batch_size > num_slices:
                raise Exception(
                    f"Batch size {self.batch_size} is greater than the number of slices in the image {num_slices}. Data loader cannot be used."
                )

            if self.augment:
                slice_idx = np.random.choice(range(num_slices), num_slices)
                img, label = img[:, :, slice_idx], label[:, :, slice_idx]

            img_batch, label_batch = get_batch(img, label, idy, num_slices)

            if self.augment:
                img_batch, label_batch = augment_data(img_batch, label_batch)

            if len(img_batch.shape) == 3:
                img_batch = np.expand_dims(img_batch, axis=-1)
            if len(label_batch.shape) == 3:
                label_batch = np.expand_dims(label_batch, axis=-1)

            yield (
                np.transpose(img_batch, [2, 0, 1, 3]).astype(np.float32),
                np.transpose(label_batch, [2, 0, 1, 3]).astype(np.float32),
            )

            idy += self.batch_size
            if idy >= num_slices:
                idy = 0
                idx = (idx + 1) % len(self.filenames)
                if idx == 0:
                    np.random.shuffle(
                        self.filenames
                    )  # Shuffle the filenames for the next iteration

    def get_input_shape(self):
        return self.crop_dim[0], self.crop_dim[1], 1

    def get_output_shape(self):
        return self.crop_dim[0], self.crop_dim[1], 1

    def get_dataset(self):
        return self.generate_batch_from_files()

    def __len__(self):
        return (self.num_slices_per_scan * self.num_files) // self.batch_size

    def __getitem__(self, idx):
        return next(self.ds)

    def plot_samples(self):
        import matplotlib.pyplot as plt

        img, label = next(self.ds)

        def plot_slice(input_image, input_label, slice_num, subplot_idx):
            plt.axis("off")
            plt.subplot(2, 2, subplot_idx)
            plt.imshow(input_image[slice_num, :, :, 0], cmap="bone")
            plt.title(
                f"MRI, Slice #{slice_num}",
                fontdict={"fontname": settings.FONT_FAMILY},
                fontsize=settings.FONT_SIZE,
            )
            plt.axis("off")
            plt.subplot(2, 2, subplot_idx + 1)
            plt.imshow(input_label[slice_num, :, :, 0], cmap="bone")
            plt.title(
                f"Tumor, Slice #{slice_num}",
                fontdict={"fontname": settings.FONT_FAMILY},
                fontsize=settings.FONT_SIZE,
            )

        plt.figure(figsize=(10, 10), dpi=settings.DPI)
        plot_slice(img, label, self.batch_size // 2, 1)
        plot_slice(img, label, self.batch_size - 1, 3)
        plt.show()
