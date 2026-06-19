-- Run this script in the Supabase SQL Editor to add emotion detection support
-- to the recognition_history table.

-- Add the detected_emotion column (nullable, defaults to 'Neutral')
ALTER TABLE public.recognition_history
    ADD COLUMN IF NOT EXISTS detected_emotion TEXT DEFAULT 'Neutral';

-- Optional: Add a comment for documentation
COMMENT ON COLUMN public.recognition_history.detected_emotion IS
    'Facial emotion detected during sign recognition. Values: Happy, Sad, Angry, Surprise, Neutral, Fear';