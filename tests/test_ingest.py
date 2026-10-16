import json
import numpy as np
import pytest
from src.common.units import altitude_to_height, to_internal, to_metres
from src.preprocess.ingest import DEFAULT_ROOT, ingest_scenario


def test_units():
    x = np.asarray([[1., 2., 166.8]])
    np.testing.assert_allclose(to_metres(to_internal(x)), x)
    assert float(altitude_to_height(166.8)) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        to_internal(x, 0)


def test_real_scenario_row_by_row(tmp_path):
    entry = json.loads((DEFAULT_ROOT / 'index.jsonl').read_text().splitlines()[0])
    path = ingest_scenario(DEFAULT_ROOT, entry, tmp_path)
    with np.load(path, allow_pickle=False) as data:
        ids = {v: i for i, v in enumerate(data['agent_id'])}
        seen = np.zeros_like(data['valid'])
        for line in (DEFAULT_ROOT / entry['path'] / 'reference_tracks_3hz.jsonl').open():
            row = json.loads(line)
            a, t = ids[row['source_object_id']], row['frame_index']
            seen[a, t] = True
            assert data['valid'][a, t]
            expected = [*row['xyz_enu_m'][:2], row['altitude_m'] - 166.]
            np.testing.assert_allclose(data['pos'][a, t], expected, rtol=1e-6, atol=0.001)
            assert data['heading'][a, t] == pytest.approx(row['heading_rad'], abs=1e-6)
            assert data['on_ground'][a, t] == row['on_ground']
            assert data['alt_fallback'][a, t] == row['altitude_fallback_used']
            assert data['agent_type'][a] == (row['adsb_reported_type'] or '')
        np.testing.assert_array_equal(seen, data['valid'])
        assert len(ids) == entry['aircraft']
        assert int(data['valid'].sum()) == entry['reference_samples']
        assert str(data['date']) == entry['start_utc'][:10]
