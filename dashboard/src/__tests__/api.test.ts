/**
 * Tests for services/api.ts — API client functions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  fetchHealth,
  fetchStats,
  fetchAlerts,
  fetchModelHealth,
  postPredict,
} from '../services/api'

const mockJsonResponse = (
  data: unknown,
  init: { ok?: boolean; status?: number; statusText?: string } = {}
) =>
  Promise.resolve({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    json: () => Promise.resolve(data),
  } as Response)

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('fetchHealth', () => {
  it('calls /health', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ status: 'healthy' })
    )
    await fetchHealth()
    expect(spy).toHaveBeenCalledWith('/health')
  })

  it('returns parsed JSON', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ status: 'healthy', model_loaded: true })
    )
    const result = await fetchHealth()
    expect(result.status).toBe('healthy')
  })
})

describe('fetchStats', () => {
  it('uses default hours=24', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ total_alerts: 100 })
    )
    await fetchStats()
    expect(spy).toHaveBeenCalledWith('/api/stats?hours=24')
  })

  it('uses custom hours', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ total_alerts: 100 })
    )
    await fetchStats(72)
    expect(spy).toHaveBeenCalledWith('/api/stats?hours=72')
  })

  it('rejects 500 responses with status and backend detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse(
        { detail: 'Failed to fetch stats: database unavailable' },
        { ok: false, status: 500, statusText: 'Internal Server Error' }
      )
    )

    await expect(fetchStats()).rejects.toThrow(
      'Request to /api/stats?hours=24 failed with 500 Internal Server Error: Failed to fetch stats: database unavailable'
    )
  })
})

describe('fetchAlerts', () => {
  it('uses default params', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ alerts: [], total: 0 })
    )
    await fetchAlerts()
    expect(spy).toHaveBeenCalledWith('/api/alerts?limit=50&offset=0')
  })

  it('appends notification_type when provided', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ alerts: [], total: 0 })
    )
    await fetchAlerts(50, 0, 'stuck')
    expect(spy).toHaveBeenCalledWith(
      '/api/alerts?limit=50&offset=0&notification_type=stuck'
    )
  })

  it('omits notification_type when undefined', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ alerts: [], total: 0 })
    )
    await fetchAlerts(10, 5)
    expect(spy).toHaveBeenCalledWith('/api/alerts?limit=10&offset=5')
  })
})

describe('fetchModelHealth', () => {
  it('uses default hours=24', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ status: 'healthy' })
    )
    await fetchModelHealth()
    expect(spy).toHaveBeenCalledWith('/api/stats/model-health?hours=24')
  })
})

describe('postPredict', () => {
  it('sends POST with JSON body', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse({ needs_intervention: true })
    )
    const payload = {
      vehicle_id: 'v1',
      speed: 30,
      expected_speed: 35,
      road_type: 'highway',
      traffic_condition: 'light',
      construction_zone: 'none',
      notification_type: 'stuck',
      pedestrian_density: 0.3,
      object_in_path: false,
      time_since_stop: 0,
    }
    await postPredict(payload)
    expect(spy).toHaveBeenCalledWith('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  })

  it('rejects 401 responses with status and auth detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse(
        { detail: 'Invalid or missing API key' },
        { ok: false, status: 401, statusText: 'Unauthorized' }
      )
    )
    const payload = {
      vehicle_id: 'v1',
      speed: 30,
      expected_speed: 35,
      road_type: 'highway',
      traffic_condition: 'light',
      construction_zone: 'none',
      notification_type: 'stuck',
      pedestrian_density: 0.3,
      object_in_path: false,
      time_since_stop: 0,
    }

    await expect(postPredict(payload)).rejects.toThrow(
      'Request to /api/predict failed with 401 Unauthorized: Invalid or missing API key'
    )
  })

  it('rejects 422 responses with formatted validation detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      mockJsonResponse(
        {
          detail: [
            {
              loc: ['body', 'speed'],
              msg: 'Input should be greater than or equal to 0',
            },
          ],
        },
        { ok: false, status: 422, statusText: 'Unprocessable Entity' }
      )
    )
    const payload = {
      vehicle_id: 'v1',
      speed: -1,
      expected_speed: 35,
      road_type: 'highway',
      traffic_condition: 'light',
      construction_zone: 'none',
      notification_type: 'stuck',
      pedestrian_density: 0.3,
      object_in_path: false,
      time_since_stop: 0,
    }

    await expect(postPredict(payload)).rejects.toThrow(
      'Request to /api/predict failed with 422 Unprocessable Entity: body.speed: Input should be greater than or equal to 0'
    )
  })
})
