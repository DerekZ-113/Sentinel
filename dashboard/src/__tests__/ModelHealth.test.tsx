/**
 * Tests for ModelHealth component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import ModelHealth from '../components/ModelHealth'

const mockHealthData = {
  status: 'healthy',
  total_predictions: 1000,
  pct_flagged: 30.0,
  pct_suppressed: 70.0,
  avg_confidence: 0.85,
  accuracy: 0.92,
  confidence_buckets: { high: 600, medium: 300, low: 100 },
  flagged_by_type: { stuck: 100, verification_request: 200 },
  suppressed_by_type: { stuck: 300, verification_request: 400 },
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('ModelHealth', () => {
  it('shows loading state initially', () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(new Promise(() => {}))
    render(<ModelHealth />)
    expect(screen.getByText('Loading model health...')).toBeInTheDocument()
  })

  it('shows healthy status badge', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockHealthData),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('Healthy')).toBeInTheDocument()
    })
  })

  it('shows prediction count', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockHealthData),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('1,000')).toBeInTheDocument()
    })
  })

  it('shows avg confidence as percentage', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockHealthData),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('85.0%')).toBeInTheDocument()
    })
  })

  it('shows accuracy as percentage', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockHealthData),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('92.0%')).toBeInTheDocument()
    })
  })

  it('shows suppression rate', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockHealthData),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('70.0%')).toBeInTheDocument()
    })
  })

  it('handles null avg_confidence', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ ...mockHealthData, avg_confidence: null }),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('N/A')).toBeInTheDocument()
    })
  })

  it('shows degraded status', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({ ...mockHealthData, status: 'degraded' }),
      } as Response)
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('Degraded')).toBeInTheDocument()
    })
  })

  it('shows error state on fetch failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.reject(new Error('Server down'))
    )
    render(<ModelHealth />)
    await waitFor(() => {
      expect(screen.getByText('Server down')).toBeInTheDocument()
      expect(screen.getByText('Retry')).toBeInTheDocument()
    })
  })
})
