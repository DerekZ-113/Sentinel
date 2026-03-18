/**
 * Tests for SimulatePanel component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SimulatePanel from '../components/SimulatePanel'

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('SimulatePanel', () => {
  it('renders the Run Prediction button', () => {
    render(<SimulatePanel />)
    expect(screen.getByText('Run Prediction')).toBeInTheDocument()
  })

  it('shows notification type selector', () => {
    render(<SimulatePanel />)
    expect(screen.getByText('Notification Type')).toBeInTheDocument()
  })

  it('shows road type selector', () => {
    render(<SimulatePanel />)
    expect(screen.getByText('Road Type')).toBeInTheDocument()
  })

  it('shows traffic selector', () => {
    render(<SimulatePanel />)
    expect(screen.getByText('Traffic')).toBeInTheDocument()
  })

  it('shows speed slider', () => {
    render(<SimulatePanel />)
    // Speed slider label renders as "Speed: <value> mph"
    expect(screen.getByText(/Expected Speed/)).toBeInTheDocument()
  })

  it('calls API on predict click', async () => {
    const user = userEvent.setup()
    const spy = vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({
          vehicle_id: 'sim_test',
          notification_type: 'stuck',
          needs_intervention: false,
          confidence: 0.85,
          raw_score: 0.15,
          timestamp: '2024-12-01T12:00:00Z',
        }),
      } as Response)
    )

    render(<SimulatePanel />)
    await user.click(screen.getByText('Run Prediction'))

    await waitFor(() => {
      expect(spy).toHaveBeenCalled()
    })
  })

  it('shows result after prediction', async () => {
    const user = userEvent.setup()
    vi.spyOn(globalThis, 'fetch').mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve({
          vehicle_id: 'sim_test',
          notification_type: 'stuck',
          needs_intervention: false,
          confidence: 0.85,
          raw_score: 0.15,
          timestamp: '2024-12-01T12:00:00Z',
        }),
      } as Response)
    )

    render(<SimulatePanel />)
    await user.click(screen.getByText('Run Prediction'))

    await waitFor(() => {
      expect(screen.getByText(/Suppress/)).toBeInTheDocument()
    })
  })
})
