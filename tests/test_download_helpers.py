import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import download


class AlignmentTests(unittest.TestCase):
    def test_plus_minus_one_pixel_offsets_and_shapes(self):
        for height, width in ((4, 5), (1, 1)):
            for source_h in range(max(1, height - 1), height + 2):
                for source_w in range(max(1, width - 1), width + 2):
                    source = np.arange(2 * source_h * source_w, dtype=np.int16)
                    source = source.reshape(2, source_h, source_w)
                    for row_offset in (-1, 0, 1):
                        for col_offset in (-1, 0, 1):
                            aligned = download.align_aod_array(
                                source, height, width, row_offset, col_offset)
                            self.assertEqual(aligned.shape, (2, height, width))
                            expected = np.zeros((2, height, width), dtype=np.int16)
                            expected[0].fill(-32768)
                            for ty in range(height):
                                for tx in range(width):
                                    sy, sx = ty + row_offset, tx + col_offset
                                    if 0 <= sy < source_h and 0 <= sx < source_w:
                                        expected[:, ty, tx] = source[:, sy, sx]
                            np.testing.assert_array_equal(aligned, expected)

    def test_rejects_invalid_bands(self):
        with self.assertRaises(ValueError):
            download.align_aod_array(np.zeros((1, 2, 2)), 2, 2, 0, 0)

    def test_ee_and_gdal_transform_conventions(self):
        ee_tf = (10, 0, 100, 0, -10, 200)
        self.assertEqual(download.ee_transform_to_gdal(ee_tf),
                         (100, 10, 0, 200, 0, -10))

    def test_fetch_budget_stops_at_configured_limit(self):
        budget = download.FetchBudget(2)
        budget.reserve()
        budget.reserve()
        with self.assertRaises(download.FetchLimitExceeded):
            budget.reserve()

class GridAndStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_data = download.DATA
        download.DATA = Path(self.tempdir.name)

    def tearDown(self):
        download.DATA = self.original_data
        self.tempdir.cleanup()

    def test_merra_grid_uses_documented_cell_centres(self):
        self.assertEqual(download.MERRA_NATIVE_TF[2] + download.MERRA_NATIVE_TF[0] / 2, -180.0)
        self.assertEqual(download.MERRA_NATIVE_TF[5] + download.MERRA_NATIVE_TF[4] / 2, -90.0)
        self.assertEqual(download.MERRA_OUT_TF[0] + download.MERRA_OUT_TF[1] / 2, 66.875)
        self.assertEqual(download.MERRA_OUT_TF[0] + 1.5 * download.MERRA_OUT_TF[1], 67.5)
        self.assertEqual(download.MERRA_OUT_TF[3] + download.MERRA_OUT_TF[5] / 2, 5.0)
        self.assertEqual((download.MERRA_ROWS, download.MERRA_COLS), (69, 52))

    def test_zero_valued_masks_remain_data_in_xarray(self):
        store = Path(self.tempdir.name) / 'static.zarr'
        group = zarr.open_group(str(store), mode='a')
        source = np.array([[0, 1], [1, 0]], dtype='uint8')
        group.create_dataset('mask', data=source, chunks=(2, 2), fill_value=0)
        download._write_mask(group, 'mask', source)
        zarr.consolidate_metadata(str(store))
        ds = xr.open_zarr(store, consolidated=True)
        np.testing.assert_array_equal(ds.mask.values, source)
        ds.close()

    def test_zero_counts_and_completion_flags_remain_data_in_xarray(self):
        store = Path(self.tempdir.name) / 'aod.zarr'
        group = zarr.open_group(str(store), mode='a').require_group('2023')
        group.create_dataset('complete', shape=(1, 1, 1), chunks=(1, 1, 1),
                             dtype='uint8', fill_value=0)
        group['complete'][0, 0, 0] = 1
        counts = np.array([[[0, 2], [3, 0]]], dtype='uint8')
        group.create_dataset('accepted_count', data=counts, chunks=(1, 2, 2),
                             dtype='uint8', fill_value=0)
        download._ensure_zero_is_data(group, 'complete',
                                      ['time', 'chunk_y', 'chunk_x'])
        download._ensure_count_fill(group)
        zarr.consolidate_metadata(str(store))
        ds = xr.open_zarr(store, group='2023', consolidated=True)
        self.assertEqual(int(ds.complete.values[0, 0, 0]), 1)
        np.testing.assert_array_equal(ds.accepted_count.values, counts)
        ds.close()

    def test_migrates_old_merra_grid_preserving_legacy_and_xarray_opens(self):
        year = 2023
        days = 365
        old = zarr.open_group(str(download.DATA / 'merra2.zarr'), mode='a').require_group(str(year))
        old.create_dataset('totexttau', shape=(days, 68, 52), chunks=(1, 68, 52),
                           dtype='float32', fill_value=-9999)
        old['totexttau'][14, 0, 0] = 0.25
        old.create_dataset('complete', data=np.zeros(days, dtype='uint8'), chunks=(1,))
        old['complete'][14] = 1
        old.create_dataset('time', data=download._epoch_days(year, days), chunks=(days,))
        old['time'].attrs['_ARRAY_DIMENSIONS'] = ['time']
        old.create_dataset('x', data=np.arange(52) + 67.3125, chunks=(52,))
        old['x'].attrs['_ARRAY_DIMENSIONS'] = ['x']
        old.create_dataset('y', data=np.arange(68) + 5.25, chunks=(68,))
        old['y'].attrs['_ARRAY_DIMENSIONS'] = ['y']
        old.attrs['native_transform_gdal'] = [67.0, .625, 0, 5.0, 0, .5]

        grid = (2, 3, (download.RES, 0, download.XORIGIN, 0, -download.RES, download.YORIGIN))
        stores = download.init_year(year, grid[0], grid[1], grid[2])
        mg = stores[('merra2', year)]
        self.assertEqual(mg['totexttau'].shape, (days, 69, 52))
        self.assertEqual(int(mg['complete'][14]), 0)
        self.assertEqual(float(mg['totexttau_legacy_68x52'][14, 0, 0]), .25)
        self.assertEqual(int(mg['complete_legacy_68x52'][14]), 1)
        self.assertEqual(float(mg['legacy_x_68x52'][0]), 67.3125)
        self.assertEqual(float(mg['x'][0]), 66.875)
        self.assertEqual(float(mg['y'][0]), 5.0)
        self.assertEqual(mg['complete'].attrs['_ARRAY_DIMENSIONS'], ['time'])

        for store, group in (('aod', str(year)), ('era5', str(year)),
                             ('merra2', str(year))):
            zarr.consolidate_metadata(str(download.DATA / f'{store}.zarr'))
            opened = xr.open_zarr(download.DATA / f'{store}.zarr', group=group,
                                  consolidated=True)
            self.assertIn('time', opened.dims)
            if store == 'aod':
                self.assertEqual(int(opened['complete'].isel(time=14, chunk_y=0, chunk_x=0)), 0)
            opened.close()


if __name__ == '__main__':
    unittest.main()
