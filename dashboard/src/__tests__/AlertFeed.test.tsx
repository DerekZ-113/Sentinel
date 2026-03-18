/**
 * Tests for AlertFeed component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import AlertFeed from '../components/AlertFeed'

const mockAlerts = [
  {
    id: 1,
    time: '2024-12-01T12:00:00Z',
    vehicle_id: 'vehicle_001',
    notification_type: 'stuck',
    notification_subtype: null,
    needs_intervention_predicted: true,
    needs_intervention_actual: true,
    confidence: 0.92,
    speed: 0,
    road_type: 'downtown',
    traffic_condition: 'heavy',
  },
  {
    id: 2,
    time: '2024-12-01T12:01:00Z',
    vehicle_id: 'vehicle_002',
    notification_type: 'verification_request',
    notification_subtype: 'object_query',
    needs_intervention_predicted: false,
    needs_intervention_actual: false,
    confidence: 0.85,
    speed: 30,
    road_type: 'main_road',
    traffic_condition: 'moderate',
  },
  {
    id: 3,
    time: '2024-12-01T12:02:00Z',
    vehicle_id: 'vehicle_003',
    notification_type: 'emergency_vehicle_alert',
    notification_subtype: null,
    needs_intervention_predicted: true,
    needs_intervention_actual: null,
    confidence: 0.65,
    speed: 30,
    road_type: 'highway',
    traffic_condition: 'light',
  },
]

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('AlertFeed', () => {
  it('shows loading state initially', () => {
    // Mock fetch to never resolve
    vi.spyOn(globalThis, 'fetch').mockReturnValue(new Promise(() => {}))
    render(<AlertFeed />)
    expect(screen.getByText('Loading alerts...')).toBeInTheDocument()
  })

  it('renders alerts after loading', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 500, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getByText('vehicle_001')).toBeInTheDocument()
    })
  })

  it('displays vehicle_id', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 3, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getByText('vehicle_002')).toBeInTheDocument()
    })
  })

  it('shows Flag for predicted intervention', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 3, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getAllByText(/Flag/).length).toBeGreaterThan(0)
    })
  })

  it('shows Suppress for non-intervention', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 3, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getAllByText(/Suppress/).length).toBeGreaterThan(0)
    })
  })

  it('shows total count', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 500, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getByText('500 total')).toBeInTheDocument()
    })
  })

  it('shows confidence percentage', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 3, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getByText('92%')).toBeInTheDocument()
    })
  })

  it('shows dash for null actual', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 3, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      // Alert 3 has null actual — should show dashes
      const dashes = screen.getAllByText('—')
      expect(dashes.length).toBeGreaterThan(0)
    })
  })

  it('shows error state on fetch failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.reject(new Error('Network error'))
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument()
      expect(screen.getByText('Retry')).toBeInTheDocument()
    })
  })

  it('shows pagination controls', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ alerts: mockAlerts, total: 100, limit: 20, offset: 0 }),
      } as Response)
    )
    render(<AlertFeed />)
    await waitFor(() => {
      expect(screen.getByText('Previous')).toBeInTheDocument()
      expect(screen.getByText('Next')).toBeInTheDocument()
      expect(screen.getByText('Page 1 of 5')).toBeInTheDocument()
    })
  })
})
