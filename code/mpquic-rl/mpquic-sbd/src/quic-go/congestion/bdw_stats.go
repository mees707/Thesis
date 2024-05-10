package congestion

import (
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
)

// BDWStats provides estimated bandwidth statistics
type BDWStats struct {
	bandwidth       Bandwidth //SHI: bit per second
	compareWindow   [10]Bandwidth
	roundRobinIndex uint8 //SHI: resume where ended
}

// NewBDWStats makes a properly initialized BDWStats object
func NewBDWStats(bandwidth Bandwidth) *BDWStats {
	return &BDWStats{
		bandwidth: bandwidth,
	}
}

//GetBandwidth returns estimated bandwidth in Mbps
func (b *BDWStats) GetBandwidth() Bandwidth { return b.bandwidth / Bandwidth(1024) } //1048576

// UpdateBDW updates the bandwidth based on a new sample.
func (b *BDWStats) UpdateBDWOLD(sentDelta protocol.ByteCount, sentDelay time.Duration) {
	disable := false
	if !disable {
		//bit per second
		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test begin: sentDelta = %d, sentDelay = %s", sentDelta, sentDelay.String())
		// }

		bdw := Bandwidth(sentDelta) * Bandwidth(time.Second) / Bandwidth(sentDelay) * BytesPerSecond
		size := uint8(len(b.compareWindow))
		startIndex := b.roundRobinIndex
		b.compareWindow[(startIndex)%size] = bdw

		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test: now changed compareWindow[%d] = %d, bdw = %d", startIndex, b.compareWindow[(startIndex)%size], bdw)
		// }

		b.roundRobinIndex = (b.roundRobinIndex + 1) % size
		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test: increased roundRobinIndex = %d", b.roundRobinIndex)
		// }
		for i := uint8(0); i < size; i++ {
			// if utils.Debug() {
			// 	utils.Debugf("UpdateBDW In test: compareWindow = %d bps", b.compareWindow[i])
			// }
			if b.bandwidth < b.compareWindow[i] {
				b.bandwidth = b.compareWindow[i]
			}
		}
		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test: sentDelta %d, sentDelay %s, fullbandwidth %d Mbps", sentDelta, sentDelay.String(), b.bandwidth/1048576)
		// }
	}
}

func (b *BDWStats) UpdateBDW(bdw Bandwidth) {
	disable := false
	if !disable {
		//bit per second
		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test begin: sentDelta = %d, sentDelay = %s", sentDelta, sentDelay.String())
		// }

		//bdw := Bandwidth(sentDelta) * Bandwidth(time.Second) / Bandwidth(sentDelay) * BytesPerSecond
		size := uint8(len(b.compareWindow))
		startIndex := b.roundRobinIndex
		b.compareWindow[(startIndex)%size] = bdw

		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test: now changed compareWindow[%d] = %d, bdw = %d", startIndex, b.compareWindow[(startIndex)%size], bdw)
		// }

		b.roundRobinIndex = (b.roundRobinIndex + 1) % size
		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test: increased roundRobinIndex = %d", b.roundRobinIndex)
		// }
		sum := uint64(0)
		count := uint64(0)
		for i := uint8(0); i < size; i++ {
			// if utils.Debug() {
			// 	utils.Debugf("UpdateBDW In test: compareWindow = %d bps", b.compareWindow[i])
			// }

			//calculation for max
			/*
				if b.bandwidth < b.compareWindow[i] {
					b.bandwidth = b.compareWindow[i]
				}
			*/
			if b.compareWindow[i] > 0 {
				count += 1
			}
			sum += uint64(b.compareWindow[i])
		}
		if count > 0 {
			b.bandwidth = Bandwidth(sum / count)
		}
		// if utils.Debug() {
		// 	utils.Debugf("UpdateBDW In test: sentDelta %d, sentDelay %s, fullbandwidth %d Mbps", sentDelta, sentDelay.String(), b.bandwidth/1048576)
		// }
	}
}
