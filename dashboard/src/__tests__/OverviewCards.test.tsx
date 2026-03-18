/**
 * Tests for OverviewCards component.
 */

import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import OverviewCards from '../components/OverviewCards'
import type { StatsResponse } from '../services/api'

function makeStats(overrides: Partial<StatsResponse> = {}): StatsResponse {
  return {
    time_window_hours: 24,
    total_alerts: 1000,
    total_flagged: 300,
    total_suppressed: 700,
    overall_fp_rate: 0.25,
    by_type: [],
    ...overrides,
  }
}

describe('OverviewCards', () => {
  it('renders total alerts', () => {
    render(<OverviewCards stats={makeStats({ total_alerts: 1000 })} />)
    expect(screen.getByText('1,000')).toBeInTheDocument()
  })

  it('renders flagged count', () => {
    render(<OverviewCards stats={makeStats({ total_flagged: 300 })} />)
    expect(screen.getByText('300')).toBeInTheDocument()
  })

  it('renders suppressed count', () => {
    render(<OverviewCards stats={makeStats({ total_suppressed: 700 })} />)
    expect(screen.getByText('700')).toBeInTheDocument()
  })

  it('calculates suppression rate', () => {
    render(<OverviewCards stats={makeStats()} />)
    expect(screen.getByText('70.0% of alerts filtered')).toBeInTheDocument()
  })

  it('displays FP rate as percentage', () => {
    render(<OverviewCards stats={makeStats({ overall_fp_rate: 0.25 })} />)
    expect(screen.getByText('25.0%')).toBeInTheDocument()
  })

  it('shows N/A when FP rate is null', () => {
    render(<OverviewCards stats={makeStats({ overall_fp_rate: null })} />)
    expect(screen.getByText('N/A')).toBeInTheDocument()
  })

  it('shows time window in subtitle', () => {
    render(<OverviewCards stats={makeStats({ time_window_hours: 48 })} />)
    expect(screen.getByText('Last 48h')).toBeInTheDocument()
  })

  it('handles zero alerts', () => {
    render(<OverviewCards stats={makeStats({
      total_alerts: 0, total_flagged: 0, total_suppressed: 0,
    })} />)
    // Multiple "0" values rendered — just verify it doesn't crash
    const zeros = screen.getAllByText('0')
    expect(zeros.length).toBeGreaterThanOrEqual(1)
  })
})
