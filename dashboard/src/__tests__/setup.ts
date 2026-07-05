import '@testing-library/jest-dom'

// jsdom doesn't implement element scrolling; components scroll feeds to top
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = (() => {}) as typeof Element.prototype.scrollTo
}
